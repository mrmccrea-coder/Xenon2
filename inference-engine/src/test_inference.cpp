// test_inference.cpp
//
// CLI test harness for the Xenon2 Phase 1 inference engine.
//
// Usage:
//   test_inference.exe "hello, how are you?"
//   test_inference.exe "hello, how are you?" --gpu-layers 24
//   test_inference.exe "hello, how are you?" --model path\to\model.bin --vocab path\to\vocab.bin
//
// Prints the model's streamed response to stdout token-by-token as it is generated (not
// batched), then prints timing/throughput stats to stderr.
#include "xenon_inference.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace {

size_t current_working_set_bytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
    return 0;
#else
    return 0;
#endif
}

struct Args {
    std::string prompt;
    std::string model_path = "models\\rwkv-5-world-0.4B-Q4_0.bin";
    std::string vocab_path = "inference-engine\\data\\world_vocab.bin";
    uint32_t threads = 6;
    uint32_t gpu_layers = 0;
    int max_tokens = 100;
    float temperature = 0.8f;
    float top_p = 0.5f;
    bool measure_memory = false;
    bool no_stop = false;
    bool has_prompt = false;
};

bool parse_args(int argc, char ** argv, Args & out) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];

        auto next = [&](const char * flag) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for %s\n", flag);
                exit(1);
            }
            return argv[++i];
        };

        if (a == "--model") out.model_path = next("--model");
        else if (a == "--vocab") out.vocab_path = next("--vocab");
        else if (a == "--threads") out.threads = static_cast<uint32_t>(std::stoul(next("--threads")));
        else if (a == "--gpu-layers") out.gpu_layers = static_cast<uint32_t>(std::stoul(next("--gpu-layers")));
        else if (a == "--max-tokens") out.max_tokens = std::stoi(next("--max-tokens"));
        else if (a == "--temperature") out.temperature = std::stof(next("--temperature"));
        else if (a == "--top-p") out.top_p = std::stof(next("--top-p"));
        else if (a == "--measure-memory") out.measure_memory = true;
        else if (a == "--no-stop") out.no_stop = true;
        else if (!out.has_prompt) { out.prompt = a; out.has_prompt = true; }
        else { fprintf(stderr, "Unexpected argument: %s\n", a.c_str()); return false; }
    }

    return out.has_prompt;
}

// Chat-style prime, matching rwkv.cpp's own python/prompt/English-Chat.json convention, so a
// small base World model produces a coherent turn rather than an open-ended continuation.
std::string build_chat_prompt(const std::string & user_message) {
    return
        "The following is a coherent, friendly conversation between a user and Xenon, a "
        "helpful voice assistant.\n\n"
        "User: Hello Xenon, how are you doing?\n\n"
        "Xenon: Hi! I'm doing well, thanks for asking. How can I help you today?\n\n"
        "User: " + user_message + "\n\n"
        "Xenon:";
}

struct CallbackState {
    std::chrono::steady_clock::time_point gen_start;
    std::chrono::steady_clock::time_point first_token_time;
    bool got_first_token = false;
    int tokens_emitted = 0;
    std::string tail; // accumulated text, for stop-sequence detection
    bool measure_memory = false;
    bool no_stop = false;
    std::vector<std::pair<int, size_t>> mem_samples; // (token index, working set bytes)
};

bool ends_with_stop(const std::string & tail) {
    static const char * stops[] = { "\n\nUser:", "\n\nuser:" };
    for (const char * s : stops) {
        size_t len = strlen(s);
        if (tail.size() >= len && tail.compare(tail.size() - len, len, s) == 0) return true;
    }
    return false;
}

bool on_token(const char * text, uint32_t /*token_id*/, void * user_data) {
    auto * st = static_cast<CallbackState *>(user_data);

    if (text[0] != '\0') {
        if (!st->got_first_token) {
            st->first_token_time = std::chrono::steady_clock::now();
            st->got_first_token = true;
        }

        fputs(text, stdout);
        fflush(stdout);

        st->tail += text;
        if (st->tail.size() > 64) st->tail.erase(0, st->tail.size() - 64);
    }

    st->tokens_emitted++;

    if (st->measure_memory && (st->tokens_emitted % 10 == 0)) {
        st->mem_samples.emplace_back(st->tokens_emitted, current_working_set_bytes());
    }

    if (!st->no_stop && ends_with_stop(st->tail)) {
        return false; // stop generation, we hit the natural end of this turn
    }

    return true;
}

} // namespace

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        fprintf(stderr,
            "Usage: %s \"<prompt>\" [--model PATH] [--vocab PATH] [--threads N] "
            "[--gpu-layers N] [--max-tokens N] [--temperature F] [--top-p F] [--measure-memory] [--no-stop]\n",
            argc > 0 ? argv[0] : "test_inference");
        return 1;
    }

    fprintf(stderr, "xenon_inference: GPU support compiled in: %s\n", xenon_has_gpu_support() ? "yes" : "no");
    fprintf(stderr, "xenon_inference: loading model '%s' (threads=%u, gpu_layers=%u)...\n",
            args.model_path.c_str(), args.threads, args.gpu_layers);

    size_t mem_before_load = current_working_set_bytes();
    auto load_start = std::chrono::steady_clock::now();

    xenon_engine * engine = xenon_load_model(
        args.model_path.c_str(), args.vocab_path.c_str(), args.threads, args.gpu_layers
    );

    auto load_end = std::chrono::steady_clock::now();

    if (!engine) {
        fprintf(stderr, "Failed to load model: %s\n", xenon_get_last_error());
        return 1;
    }

    double load_sec = std::chrono::duration<double>(load_end - load_start).count();
    size_t mem_after_load = current_working_set_bytes();

    fprintf(stderr, "xenon_inference: model loaded in %.3f sec. n_layer=%zu state_len=%zu (fixed-size state, %.2f MB)\n",
            load_sec, xenon_get_n_layer(engine), xenon_get_state_len(engine),
            xenon_get_state_len(engine) * sizeof(float) / (1024.0 * 1024.0));
    fprintf(stderr, "xenon_inference: working set after load: %.1f MB (delta from pre-load: %.1f MB)\n",
            mem_after_load / (1024.0 * 1024.0),
            (double)(mem_after_load - mem_before_load) / (1024.0 * 1024.0));

    std::string full_prompt = build_chat_prompt(args.prompt);

    CallbackState st;
    st.measure_memory = args.measure_memory;
    st.no_stop = args.no_stop;
    st.gen_start = std::chrono::steady_clock::now();

    printf("%s", args.prompt.c_str()); // echo user prompt to stdout for readability
    fflush(stdout);

    xenon_status status = xenon_generate(
        engine, full_prompt.c_str(), args.max_tokens, args.temperature, args.top_p, on_token, &st
    );

    auto gen_end = std::chrono::steady_clock::now();
    printf("\n");

    if (status != XENON_OK) {
        fprintf(stderr, "Generation error: %s\n", xenon_get_last_error());
        xenon_free_engine(engine);
        return 1;
    }

    double total_sec = std::chrono::duration<double>(gen_end - st.gen_start).count();
    double ttft_sec = st.got_first_token
        ? std::chrono::duration<double>(st.first_token_time - st.gen_start).count()
        : -1.0;
    double tok_per_sec = st.tokens_emitted > 0 ? st.tokens_emitted / total_sec : 0.0;

    fprintf(stderr, "\n--- stats ---\n");
    fprintf(stderr, "tokens generated: %d\n", st.tokens_emitted);
    fprintf(stderr, "time to first token: %.3f sec\n", ttft_sec);
    fprintf(stderr, "total generation time: %.3f sec\n", total_sec);
    fprintf(stderr, "throughput: %.2f tokens/sec\n", tok_per_sec);

    if (args.measure_memory) {
        fprintf(stderr, "\n--- memory samples (token_index, working_set_MB) ---\n");
        for (auto & [idx, bytes] : st.mem_samples) {
            fprintf(stderr, "%4d  %.2f MB\n", idx, bytes / (1024.0 * 1024.0));
        }
    }

    xenon_free_engine(engine);
    return 0;
}
