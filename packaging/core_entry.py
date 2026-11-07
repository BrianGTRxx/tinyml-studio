"""Script de entrada que congela PyInstaller (ver core.spec). Toda la lógica vive en el paquete."""

from tinyml_studio.core_main import main

if __name__ == "__main__":
    raise SystemExit(main())
