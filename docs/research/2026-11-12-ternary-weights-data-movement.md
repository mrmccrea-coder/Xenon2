# Note: Ternary Weights and Data Movement (2026-11-12)

This comes from the Computer Architecture Lab experiments run on the same i7-12850HX laptop used for Xenon2. Nothing in Xenon2 has been changed; this records an idea worth testing later.

## The idea

- **Established fact:** the transistors in this CPU switch between two states. Software can't make them hold three.
- **What software can do:** pack non-binary values into binary storage.
  - A ternary value (−1, 0 or +1) carries about 1.58 bits of information.
  - Five of them fit in one byte, because 3⁵ = 243, which is under 256.
  - AI models with weights limited to {−1, 0, +1} already exist; one example is Microsoft Research's BitNet b1.58.
- **Why that matters:** a 16-bit weight shrinks to under 2 bits, so about 10× less data has to move from memory. If data movement is the bottleneck, that can turn into real speed.
- **The open question:** does the time spent unpacking those values eat the gain? That has to be measured, not assumed.

## Why it's relevant to Xenon2

- **Xenon2's CPU inference already looks memory-bandwidth-bound** (measured 2026-11-03):
  - 0.4B = 38 tok/s, 1.5B = 11 tok/s, 2.9B = 6 tok/s.
  - Going from 4 to 16 threads barely helps. That is the typical sign of waiting on memory rather than on arithmetic.
- **The lab measured the memory itself** (2026-11-12, same laptop):
  - One 32 GB DDR5 module running at 4000 MT/s, so the theoretical peak is about 32 GB/s.
  - One core reads RAM in order at about 18 GB/s.
  - One random read from RAM takes about 109 ns, roughly 450 core cycles.
  - Reading in order delivers about 250× more useful data per second than reading randomly.
- **Rough estimate (inference, not measured):**
  - Generating one token reads every weight once.
  - The 1.5B model at Q5_1 (about 6 bits per weight) is about 1.1 GB of weights.
  - 11 tok/s × 1.1 GB ≈ 12 GB/s, which is close to the measured single-core read speed. That fits the bandwidth-bound picture.
- **The ceiling, as a hypothesis:**
  - At about 1.58–2 bits per weight, the same model would be about 0.3–0.4 GB.
  - If unpacking can be hidden behind memory access, the ceiling for CPU decoding would be roughly 3× faster than Q5_1.

## Caveats before anyone acts on this

- **Ternary has to be trained in.** BitNet-style models are trained (or quantisation-aware fine-tuned) as ternary from the start. Rounding an existing model's weights to {−1, 0, +1} after training typically wrecks output quality.
  - The current RWKV7 G1 models are not ternary-trained.
  - Using this would need a ternary RWKV model (none is known to exist) or a BitNet-family model, which means a different backend.
- **The packing choice is a trade-off:**
  - 5 values per byte (base 3) is denser, but slower to unpack.
  - 4 values per byte (2 bits each) is simpler and faster to unpack, but uses 26% more space.
  - Which wins on this CPU is an experiment, not a guess.
- **The GPU path behaves differently.** On the A1000, launch and transfer overhead already dominated for small models, so this note is mainly about CPU decoding.

## Planned next step (in the lab, not in Xenon2)

Measure a dot product (the core operation of model inference) on the same data stored four ways, at each cache level and from RAM:
- 32-bit floats
- 8-bit integers
- packed 4-bit values
- packed ternary

The result tells us whether a smaller representation actually turns into more useful work per second on this hardware. Only after that would a Xenon2 change be worth considering.
