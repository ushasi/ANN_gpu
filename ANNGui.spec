# -*- mode: python -*-

block_cipher = None


a = Analysis(['ANNGui.py'],
             pathex=['E:\\IIT Bombay\\IITB\\Sem2\\Machine Learning\\Project'],
             binaries=[],
             datas=[('IITB.png', '.'), ('CSRE.ico', '.')],
             hiddenimports=['pandas._libs.tslibs.timedeltas', 'scipy._lib.messagestream'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='ANNGui',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True , icon='CSRE.ico')
