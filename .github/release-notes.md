First pre-release of TinyML Studio. **Early software: please read before installing.**

**What has been tested**
1. **Windows 11: tested by the author.** The installer (`.exe`) installs and runs. It was tried on a computer that already had a previous setup, so the very first run on a brand new account (where the app creates its own Python) is **not verified yet**.
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
