Second pre-release of TinyML Studio. **Early software: please read before installing.**

**What changed since v0.0.1**
1. **Fixed:** "Check for updates" failed with `net::ERR_BLOCKED_BY_CLIENT`. The app's own request filter was blocking the check. It now works. It still contacts only `api.github.com` (the public releases page) and sends nothing of yours.
2. **Documentation:** the README has a new section that lists the models, techniques and export formats.
3. The version number is now 0.0.2.

If you have v0.0.1 installed, run this installer over it: it replaces the program and keeps your settings. You do not need to uninstall first.

**What has been tested**
1. **Windows 11: tested by the author.** The installer (`.exe`) installs and runs; the update check fix was verified in Electron against the public repository. The v0.0.2 installers themselves have not been tried yet, and the very first run on a brand new account (where the app creates its own Python) is **not verified yet**.
2. **Linux (`.AppImage` and `.deb`): built but NOT tested yet.** They were built automatically and have not been opened on a real Linux system. If you try them, please tell us what happens.
3. Verified on a real ESP32-S3 with two small examples; other boards are not verified. See the README.

**Things to know**
1. The installers are **not signed**. Windows SmartScreen will warn: choose "More info", then "Run anyway".
2. On first run the app creates its own private Python environment and downloads what you choose (scikit-learn, Keras, PyTorch...), so it needs an internet connection that first time. Keras or PyTorch with CUDA add about 2 GB each.

**Files**
1. Windows 10/11: `TinyML-Studio-raw-Setup-*.exe` (installer) or `TinyML-Studio-raw-*-win-x64.zip` (portable, no install).
2. Linux: `.AppImage` (run `chmod +x` on it, then double click) or `.deb`.
3. `SHA256SUMS.txt`: check the download with it.

License: GPL-3.0. Developed with the assistance of Claude (Anthropic).
