# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules

datas = [('data/card.json', 'data'), ('data/card_popularity_weights.json', 'data'), ('data/classes.yaml', 'data')]
hiddenimports = ['ultralytics', 'cv2', 'torch', 'torchvision', 'PIL', 'yaml', 'mss', 'tkinter', 'numpy']
datas += collect_data_files('ultralytics')
hiddenimports += collect_submodules('ultralytics')


a = Analysis(
    ['/root/FaBCode/fab_detector_app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='FaBCardDetector',
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
    icon='NONE',
)
