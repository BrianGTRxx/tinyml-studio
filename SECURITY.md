# Security policy

## Supported versions
TinyML Studio is an early pre-release (0.0.x). Only the **latest release** receives fixes.

## Reporting a vulnerability
Please **do not open a public issue** for a security problem. Report it privately instead:

1. Go to the **Security** tab of this repository.
2. Choose **Report a vulnerability** (GitHub private vulnerability reporting).
3. Describe what you found, the version, your operating system, and the steps to reproduce it.

Do not attach private data, credentials or recordings. A short description or a synthetic example is enough.

I will answer on a best effort basis, normally within about a week, and I will tell you when it is fixed. You are welcome to be credited in the release notes if you wish.

## What is in scope
1. The local server that the desktop app starts (authentication, access to folders and files).
2. The desktop application (the Electron shell, the bridge between the window and the core, the update notice).
3. The installers and the way the app creates its Python environment.
4. The code and files the app generates (C, TFLite, ONNX, Arduino sketches).

## What is not a vulnerability
1. The Windows SmartScreen warning: the installers are not signed yet, and this is documented.
2. Problems that need the attacker to already control your computer or your user account.
3. Accuracy of a trained model: please open a normal issue for that.

## What the app does and does not do
1. It runs entirely on your computer. It makes no network request unless you turn on the optional notice of new versions (it asks GitHub for the latest release; nothing about you or your files is sent) or you choose to install Python packages.
2. It never installs or runs a hardware driver by itself.
