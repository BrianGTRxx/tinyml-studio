Third pre-release of TinyML Studio. **Early software: please read before installing.**

**What changed since v0.0.2**
1. **New:** update from inside the app (Windows installed copy). When "Check for updates" finds a newer version, the new **Update now** button downloads the installer from the official release of this project, checks its SHA-256 against the list published with it, and opens it. Nothing is installed until you confirm in the installer. The portable `.zip` and Linux still open the download page.
2. The installers are still **not signed**, so Windows may warn you on every update. The SHA-256 check protects against a damaged download; it cannot protect against someone who takes over the GitHub account.

**Earlier (v0.0.2)**
1. Fixed: "Check for updates" failed with `net::ERR_BLOCKED_BY_CLIENT`.
2. The README has a section that lists the models, techniques and export formats.

If you have v0.0.2 installed, run this installer over it once: it replaces the program and keeps your settings. From then on, updates can be done with the **Update now** button. You do not need to uninstall first.

**What has been tested**
1. **Windows 11: tested by the author.** The installer (`.exe`) installs and runs; the download and checksum step of the updater was verified in Electron against the real v0.0.2 release. The **full cycle (installing an update from inside the app) has not been tried yet**: v0.0.3 is the first version that has it, so it can only be tried once a newer one exists. The v0.0.3 installers themselves have not been tried yet, and the very first run on a brand new account (where the app creates its own Python) is **not verified yet**.
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
