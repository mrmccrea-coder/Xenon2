# Image regeneration and editing tests — 2026-11-04

Hardware: RTX A1000 Laptop (4 GB), i7-12850HX, 32 GB RAM. Engine: stable-diffusion.cpp
`master-1047-984eb06` (Vulkan), `--max-vram 1.0 --offload-to-cpu --vae-tiling`, with the
sdxl-vae-fp16-fix decoder. Test pictures and masks were kept in the session scratchpad, not
the repo.

## 1. Regenerate (same prompt, new seed) — works as designed

Prompt: *"a red vintage pickup truck parked beside a calm mountain lake at sunset, autumn trees,
golden light"*, 768 px, original + 3 regenerates, ~28 s each.

| Detail | Original | Regen 1 | Regen 2 | Regen 3 |
|---|---|---|---|---|
| red truck | yes | yes | yes | yes |
| vintage pickup | yes (60s) | ~ (70s look) | yes | yes (40s, whitewalls) |
| parked *beside* the lake | **no** (in the water) | **no** (in shallows) | **no** (floating) | yes (on shore) |
| calm lake, reflections | yes | yes | yes | yes |
| mountains | yes | yes | yes | ~ hills |
| sunset / autumn trees | yes | yes | yes | yes |
| golden light | yes | yes | ~ pink-red | ~ pink |

- Regenerate draws a **completely new picture** every time (new seed): subjects and colours stay,
  layout changes.
- Weakness: **spatial relationships** ("beside") — 3 of 4 put the truck in or on the water.
- Regenerate **replaces** the picture in the chat; earlier versions survive only as files in
  `%APPDATA%\com.xenon2.app\images`. Idea: keep every version (1 of 4 ‹ ›).

## 2. "Keep everything, add a camper trailer" to regen 3

| Approach | Rest kept? | Trailer added? | Notes |
|---|---|---|---|
| img2img, SDXL-Lightning, strength 0.45 / 0.6 | yes | **no** | 12–14 s |
| img2img, strength 0.75 | no (new scene) | **no** (just a rack on the bed) | 15 s |
| Mask + SDXL-Lightning (not an inpaint model), right strip / wide box | yes, pixel-identical | **no** (more trees); wide box left a visible seam | 15 s |
| **SDXL inpainting 0.1** (`sdxl_inpainting_0.1.safetensors`, 6.94 GB, sha256 `fe1b97fe…13fa8a`), scene prompt, cfg 4 and 2, 24 steps euler_a | yes | **no** (filled with road/lake) | ~100 s; engine reports "Version: SDXL Inpaint" |
| same model, object-only prompt "a vintage camper trailer…", cfg 7, ellipse mask | yes | **no** | ~97 s |
| same, cfg 7, big box mask | mostly | **no** — painted the *truck's rear* red and white instead | ~96 s |
| same, cfg 6, scene+object prompt | — | — | failed: out of GPU memory (see 4) |

- A **fresh** Lightning picture asking for *"a red vintage 1940s pickup towing a small white
  vintage camper trailer… beside a calm lake at sunset"* (1024×768) came out **very well**:
  the model can draw truck + trailer, it just won't insert one into this existing picture.
- Likely reason: in regen 3 the truck fills the frame and there is almost no ground behind it
  where a trailer could stand, so inpainting continues the surroundings. **Not yet tested:**
  painting genuinely *open* ground beside the subject, which is how brush-to-edit would be
  used. That test was set up (wide scene with an empty clearing) but stopped before it ran.

## 3. Engine notes

- `sd-cli` supports `-i/--init-img`, `--mask`, `--strength`. SDXL inpainting checkpoints work
  in the CLI; the maintainers say they need **low CFG** (stable-diffusion.cpp #1309, closed).
  #1406 (mask ignored) is about the **server** API, not the CLI.
- The inpainting model is full SDXL at fp16 (6.9 GB): ~4× slower than Lightning (24 steps,
  ~100 s at 768 px) and near the limit of a 4 GB card.

## 4. Gotcha: Xenon2 open = image engine out of memory

Several runs failed with `unet segment 5/17 ... failed during weight preparation` /
`cannot make enough memory available on Vulkan1`. Cause: **the Xenon2 app was open** with its
chat model holding ~3.2 GB of the 4 GB. Inside the app this can't happen (image jobs unload the
chat model first); it only bites when running `sd-cli` by hand alongside the app.

## State of brush-to-edit

- Backend done and unit-tested, **not exposed in the UI**: `edit_image` command (source must be
  one of the app's own PNGs; mask from the editor as a PNG data URL, saved beside the images
  for Regenerate; picture size kept), `image_status.canEdit`, saved fields
  `editSource`/`editMask`. Settings in `image.rs` (`EDIT_STEPS = 24`, `EDIT_CFG = "4.0"`)
  are provisional.
- Still to do: the brush editor UI, then the open-ground test above, several seeds, and tuning
  (try a Lightning LoRA or a smaller quantized inpaint model for speed and memory).
