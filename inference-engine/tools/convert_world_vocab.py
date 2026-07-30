#!/usr/bin/env python3
"""
Converts rwkv.cpp's World tokenizer vocab file (python/rwkv_cpp/rwkv_vocab_v20230424.txt,
a text file of `<id> <python repr of str-or-bytes token> <byte length>` lines, parsed with
Python's eval()) into a small binary file that the C++ inference engine can parse trivially,
without needing to reimplement Python literal-eval escape-sequence semantics in C++.

Binary format (little-endian):
  uint32   num_entries
  repeated num_entries times:
    uint32   token_id
    uint16   byte_length
    bytes[byte_length]  raw token bytes (as they should be emitted/matched during
                         encode/decode -- this is the exact byte string produced by
                         evaluating the Python repr in the source vocab file)

Usage:
  python convert_world_vocab.py <path/to/rwkv_vocab_v20230424.txt> <output.bin>
"""

import sys
import struct


def convert(src_path: str, dst_path: str) -> None:
    entries = []

    with open(src_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip('\n')
        if not line:
            continue

        idx_str, rest = line.split(' ', 1)
        idx = int(idx_str)

        # rest is `<repr> <length>` -- repr itself may contain spaces, so split from the right.
        repr_str, length_str = rest.rsplit(' ', 1)
        expected_len = int(length_str)

        value = eval(repr_str)
        token_bytes = value.encode('utf-8') if isinstance(value, str) else value
        assert isinstance(token_bytes, (bytes, bytearray))
        assert len(token_bytes) == expected_len, (
            f'length mismatch for id {idx}: got {len(token_bytes)}, expected {expected_len}'
        )

        entries.append((idx, token_bytes))

    with open(dst_path, 'wb') as out:
        out.write(struct.pack('<I', len(entries)))
        for idx, token_bytes in entries:
            out.write(struct.pack('<IH', idx, len(token_bytes)))
            out.write(token_bytes)

    print(f'Wrote {len(entries)} vocab entries to {dst_path}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'Usage: python {sys.argv[0]} <src rwkv_vocab_*.txt> <dst .bin>')
        sys.exit(1)

    convert(sys.argv[1], sys.argv[2])
