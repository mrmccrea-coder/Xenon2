# Xenon2

A private AI chat assistant that runs entirely on your own Windows PC. No account, no internet
connection needed after setup, and nothing you type leaves your computer.

It runs the open [RWKV-7 "G1"](https://huggingface.co/BlinkDL/rwkv7-g1) language models
locally, with a built-in calculator for exact maths and unit conversions.

## Try it (Windows 10/11, 64-bit)

1. Download **`Xenon2_0.2.0_x64-setup.exe`** from the
   [latest release](https://github.com/mrmccrea-coder/Xenon2/releases/latest) and run it.
   It installs for your user only (no admin rights needed) and adds Xenon2 to the Start menu.
2. **Windows may warn you.** The app isn't code-signed yet, so you may see
   *"Windows protected your PC"* - click **More info → Run anyway**. Microsoft Defender may also
   flag it as a false positive (it's an unsigned, newly built program); if it removes the file,
   you can restore it from *Windows Security → Protection history*. Only do this if you
   downloaded it from the link above.
3. On first launch Xenon2 opens **Set up models**. Pick one and it downloads (once):

   | Model | Download | Best for |
   |---|---|---|
   | Fast | 525 MB | older or low-memory PCs; quick but makes more mistakes |
   | Balanced | 1.5 GB | most PCs |
   | Quality | 2.75 GB | the best replies; wants 8 GB+ of RAM, ideally an NVIDIA graphics card |

   The setup screen recommends one for your computer. Downloads resume if interrupted and are
   checked against a checksum before use.
4. Start chatting.

**Speed:** with an NVIDIA graphics card the model runs on it and replies quickly. Without one it
runs on the processor - slower, but it works.

## What to try

- Ordinary questions, writing ("write a short story about..."), planning, explanations.
- Maths and conversions - *"How many meters are in 5 yards?"*, *"18% tip on $45"*, *"98.6 F to C"* -
  answered exactly by the built-in calculator.
- Long conversations: ask follow-ups like *"tell me more"* or *"continue the story"*.
- **Goldfish / Sloth** (under the chat box): Goldfish remembers only the current chat; Sloth
  remembers facts about you across chats.
- **Reply length** (above the chat): *Auto* sizes replies to what you ask; you can set it yourself.
- The square **Stop** button ends a reply early.

## Known limits of this test release

- **Chat only.** Image generation and voice exist in the code but aren't packaged in the
  installer yet, so those buttons don't appear.
- These are small models: replies can be confidently wrong, especially on facts. Check anything
  important.
- Windows only for now.

## Reporting problems

Please open an [issue](https://github.com/mrmccrea-coder/Xenon2/issues) with what you typed, what
happened, which model you picked, and whether your PC has an NVIDIA graphics card. The status
in the top-right corner of the app ("Running on graphics card" / "Running on processor") helps.

## For developers

- `app/` - the desktop app (Tauri 2 + Vue 3); build notes in [app/README.md](app/README.md).
  The installer is built with `app/scripts/build_installer.ps1`.
- `inference-engine/` - the C++ engine wrapping rwkv.cpp; see
  [inference-engine/README.md](inference-engine/README.md).
- `voice-pipeline/` - the optional local voice pipeline (not in the installer yet).

The chat models are hosted in the [`models-v1` release](https://github.com/mrmccrea-coder/Xenon2/releases/tag/models-v1)
(BlinkDL's RWKV-7 G1, Apache-2.0).
