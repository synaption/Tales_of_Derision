# PyInstaller spec for the Windows executable build.
# Build from the repository root with:
#   pyinstaller --clean --noconfirm packaging/tales_of_derision_windows.spec
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path.cwd()


def tree_data(source: str, dest: str):
    return [(str(ROOT / source), dest)]


pygame_datas, pygame_binaries, pygame_hiddenimports = collect_all("pygame")

datas = [
    *tree_data("audio", "audio"),
    *tree_data("gfx", "gfx"),
    *tree_data("src/data", "src/data"),
    *pygame_datas,
]

block_cipher = None

a = Analysis(
    [str(ROOT / "src" / "main.py")],
    pathex=[str(ROOT / "src")],
    binaries=pygame_binaries,
    datas=datas,
    hiddenimports=pygame_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="TalesOfDerision",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
