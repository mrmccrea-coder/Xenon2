// world_tokenizer.h
//
// C++ re-implementation of RWKV's "World" tokenizer (a byte-level greedy-longest-match trie
// tokenizer), matching the reference Python implementation in
// rwkv.cpp/python/rwkv_cpp/rwkv_world_tokenizer.py. Loads its vocabulary from a small binary
// file produced by inference-engine/tools/convert_world_vocab.py, rather than re-parsing
// Python `repr()` literals in C++.
//
// World models use n_vocab = 65536 and REQUIRE this tokenizer (as opposed to older Pile/Raven
// models, which use the 20B BPE tokenizer and n_vocab = 50277). Using the wrong tokenizer for a
// model produces garbage output without any hard error, so xenon_inference validates
// n_vocab against the loaded vocab file's entry count on load.
#ifndef WORLD_TOKENIZER_H
#define WORLD_TOKENIZER_H

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class WorldTokenizer {
public:
    // Loads the binary vocab file. Throws std::runtime_error on failure.
    explicit WorldTokenizer(const std::string & vocab_bin_path);

    // Total number of vocab entries loaded (should match the model's n_vocab).
    size_t vocab_size() const { return decode_table_.size(); }

    // Greedy longest-match encode of UTF-8 text into token ids.
    std::vector<uint32_t> encode(const std::string & text) const;

    // Raw bytes for a single token id. Returns an empty string for unknown/unassigned ids.
    const std::string & decode_token(uint32_t id) const;

private:
    struct TrieNode {
        std::array<int32_t, 256> next; // index into nodes_ array, or -1
        int32_t token_id = -1;         // vocab id ending at this node, or -1

        TrieNode() { next.fill(-1); }
    };

    void insert(const std::string & bytes, uint32_t id);

    std::vector<TrieNode> nodes_;               // trie for encode()
    std::vector<std::string> decode_table_;      // id -> raw bytes, for decode()
};

#endif // WORLD_TOKENIZER_H
