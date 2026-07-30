#include "world_tokenizer.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

WorldTokenizer::WorldTokenizer(const std::string & vocab_bin_path) {
    FILE * f = fopen(vocab_bin_path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("WorldTokenizer: failed to open vocab file: " + vocab_bin_path);
    }

    uint32_t num_entries = 0;
    if (fread(&num_entries, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        throw std::runtime_error("WorldTokenizer: failed to read entry count from " + vocab_bin_path);
    }

    // Root node.
    nodes_.emplace_back();

    uint32_t max_id = 0;

    struct RawEntry { uint32_t id; std::string bytes; };
    std::vector<RawEntry> entries;
    entries.reserve(num_entries);

    for (uint32_t i = 0; i < num_entries; i++) {
        uint32_t id;
        uint16_t len;

        if (fread(&id, sizeof(uint32_t), 1, f) != 1 || fread(&len, sizeof(uint16_t), 1, f) != 1) {
            fclose(f);
            throw std::runtime_error("WorldTokenizer: truncated vocab file: " + vocab_bin_path);
        }

        std::string bytes(len, '\0');
        if (len > 0 && fread(&bytes[0], 1, len, f) != len) {
            fclose(f);
            throw std::runtime_error("WorldTokenizer: truncated token bytes in: " + vocab_bin_path);
        }

        if (id > max_id) max_id = id;
        entries.push_back({id, std::move(bytes)});
    }

    fclose(f);

    decode_table_.assign(static_cast<size_t>(max_id) + 1, std::string());

    for (auto & e : entries) {
        decode_table_[e.id] = e.bytes;
        insert(e.bytes, e.id);
    }
}

void WorldTokenizer::insert(const std::string & bytes, uint32_t id) {
    int32_t cur = 0; // root

    for (unsigned char ch : bytes) {
        int32_t next = nodes_[cur].next[ch];
        if (next < 0) {
            nodes_.emplace_back();
            next = static_cast<int32_t>(nodes_.size()) - 1;
            nodes_[cur].next[ch] = next;
        }
        cur = next;
    }

    nodes_[cur].token_id = static_cast<int32_t>(id);
}

std::vector<uint32_t> WorldTokenizer::encode(const std::string & text) const {
    std::vector<uint32_t> out;

    size_t i = 0;
    const size_t n = text.size();

    while (i < n) {
        int32_t cur = 0; // root
        int32_t best_id = -1;
        size_t best_len = 0;
        size_t j = i;

        while (j < n) {
            unsigned char ch = static_cast<unsigned char>(text[j]);
            int32_t next = nodes_[cur].next[ch];
            if (next < 0) break;

            cur = next;
            j++;

            if (nodes_[cur].token_id >= 0) {
                best_id = nodes_[cur].token_id;
                best_len = j - i;
            }
        }

        if (best_id < 0) {
            // Should not happen: the World vocab includes all 256 single bytes, so any byte
            // matches at least a length-1 token. Guard against corrupt vocab data anyway.
            throw std::runtime_error("WorldTokenizer: no matching token for byte at offset " + std::to_string(i));
        }

        out.push_back(static_cast<uint32_t>(best_id));
        i += best_len;
    }

    return out;
}

const std::string & WorldTokenizer::decode_token(uint32_t id) const {
    static const std::string empty;
    if (id >= decode_table_.size()) return empty;
    return decode_table_[id];
}
