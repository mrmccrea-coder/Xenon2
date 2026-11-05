# Xenon2 Test: Microsoft Store Compliance

Every change on the `xenon2-test` branch must keep these rules true. Before each Store
submission, run the checklist at the bottom. If a change would break a rule, stop and raise it
with the owner. Don't work around it.

Status key: **OK** = met now · **OPEN** = must be done before submission · **BLOCKER** = owner decision needed.

## Scope

- Branch `xenon2-test`. Product name "Xenon2 Test", identifier `com.xenon2.test`.
- Distribution: Microsoft Store, **Private audience** only, until the owner approves going public.
- The USB-portable build is **on hold**. Don't merge portable-only changes into this branch, and
  don't merge Store-only changes into `master` without the owner's approval.
- Ships **chat + image generation**, with all three chat models and the SDXL Lightning image
  model **inside the package** (about 8.4 GB). Setup is automatic: install, open, use. Voice is not
  packaged. Brush-to-edit (inpainting) is not packaged until it is in the UI.

## Rules

### C-1 Store listing and policy (Microsoft Store Policies 10.x)
- The listing must describe what the app really does: local AI chat, a one-time model download,
  and replies that can be wrong. No claims about features that aren't shipped (voice, images). **OPEN**
- Age rating questionnaire completed. The app produces unfiltered AI text, so answer the
  generated-content questions honestly. **OPEN**
- Generative AI apps: the Store expects you to make reasonable efforts against harmful output and
  to give users a way to report it. Minimum: a visible "AI can be wrong" note and a "Report a
  problem" link in the app. **OPEN** (see Big Issues #4)

### C-2 Privacy (Policy 10.5)
- A public privacy policy URL is entered in Partner Center → Properties. Text: `store/PRIVACY_POLICY.md`,
  hosted on Google Sites. **OPEN**
- The same link is reachable inside the app (About/Settings). **OPEN**
- No new network calls, analytics, telemetry or crash reporting without updating the policy
  *first*. The only allowed network use is the model download from the hosts in `model_catalog.json`. **OK**
- Any change to what is stored (chats, Sloth memory) means re-checking the policy text.

### C-3 MSIX file system
- The install folder is read-only. Store builds use `--features store`, which keeps all writes
  in the per-user app data folder (`app/src-tauri/src/paths.rs`). **OK**
- Never write beside the exe, to `Program Files`, or to the registry outside the package.
- Uninstall must remove everything (MSIX does this for app data automatically). **OK**

### C-4 No downloaded code (Policy 10.2.2)
- The app may download **data** (model weights) only. It must never download or run executables,
  DLLs or scripts after install. All DLLs ship inside the package. **OK**
- The `nvidia-smi` hardware probe runs a tool the system already has. It is allowed but must stay
  optional (failure = run on CPU). **OK**

### C-5 Capabilities
- `runFullTrust`: needed for Tauri (Win32). Partner Center asks for a justification. Use:
  "Desktop app built with Tauri; runs a local AI model through native libraries." **OPEN** (enter at submission)
- `internetClient`: kept only for the optional extra-model download screen. Models are bundled, so
  the app works with no internet at all. Remove it if that screen is hidden in Store builds. **OPEN**
- No other capabilities without updating this file.

### C-6 Microphone (only if voice ships)
- Add the `microphone` device capability, a clear in-app prompt, and a privacy policy section
  *before* voice ships. Voice also needs its Python pipeline packaged without downloaded code (C-4). Not in scope now.

### C-7 Third-party licences
- RWKV-7 G1 models: Apache-2.0. Redistribution allowed with attribution. **OK** (add a notice, see C-8)
- NVIDIA cuBLAS DLLs: redistributable under the CUDA EULA when shipped inside an application.
  The attribution NVIDIA requires must be included. **OPEN**
- MSVC runtime DLLs: redistributable. **OK**
- Rust/npm dependencies: generate a third-party notices file (`cargo about` / `license-checker`). **OPEN**

- SDXL Lightning (ByteDance) / SDXL base: CreativeML **OpenRAIL++-M**. Redistribution allowed,
  but the licence text and its **use restrictions must be passed on to users** (in the EULA/notices). **OPEN**
- SDXL VAE fp16-fix: MIT. stable-diffusion.cpp / ggml: MIT. Include their notices. **OPEN**

### C-8 Notices and terms
- A "Licences / About" screen or bundled `THIRD_PARTY_NOTICES.txt`. **OPEN**
- Beta terms (testing only, no redistribution) shown in the listing or on first run. **OPEN**

### C-9 Security
- `tauri.conf.json` has `"csp": null`. The Store doesn't require a CSP, but this app renders
  model text as HTML/markdown, so set a restrictive CSP before public release. **OPEN** (needs testing so it doesn't break the UI)

### C-10 Technical certification (WACK)
- Package passes the Windows App Certification Kit locally before upload. **OPEN**
- App launches, doesn't crash with no internet, and recovers if a download is cancelled. **OPEN**
- Version format `a.b.c.0`, raised on every upload. `build-msix.ps1` enforces the format. **OK**
- Identity values (Name, Publisher, PublisherDisplayName) copied from Partner Center, never
  the local placeholders. **OPEN**

## Big Issues (owner decisions)

1. **RESOLVED — Model downloads pointed at a private repo.** All models now ship inside the
   package, so nothing is downloaded. The old download screen still points at the private repo.
   Hide it in Store builds or leave it unused.
2. **Package size: about 8.4 GB.** Fine for the Store (limit about 25 GB), but slow for testers on
   weak connections. Later options: Store optional packages, or a CPU-only build without the CUDA files.
3. **Name.** "Xenon2" must be free in Partner Center. "Xenon" is a common word and may already be
   taken or trademarked in software. Reserve the name early and have a fallback ready.
4. **Generative AI content.** The small models can produce wrong or inappropriate text with no
   filter. Microsoft may want a content safeguard and a report path. Decide how far to go
   (disclaimer + report link is the minimum).
5. **Image generation raises the content-policy bar.** Unfiltered SDXL can produce sexual or
   violent images. The Store treats that much more strictly than text, and the age rating will
   likely go up. A safety filter or prompt blocklist is probably needed before certification.
   This is the most likely reason for a rejection.
6. **Partner Center account** must exist before the real identity values can go in the manifest.

## Pre-submission checklist

- [ ] Big Issues 1, 3, 5 resolved
- [ ] Privacy policy live on Google Sites, URL in Partner Center and in-app
- [ ] Identity values from Partner Center passed to `build-msix.ps1`
- [ ] Version bumped (`a.b.c.0`)
- [ ] `.\store\build-msix.ps1` succeeds; WACK passes
- [ ] Clean-PC test: install, download a model, chat, go offline, chat, uninstall
- [ ] Third-party notices bundled; beta terms visible
- [ ] Listing text matches what ships; age rating done; runFullTrust justification entered
- [ ] Visibility = Private audience, tester group set
