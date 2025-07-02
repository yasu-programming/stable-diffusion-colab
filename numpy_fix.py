#!/usr/bin/env python3
"""
NumPy 2.0互換性問題修正スクリプト
"""

import subprocess
import sys

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🔧 NumPy 2.0互換性問題修正中...")
    print("=" * 50)
    
    # 現在のNumPyバージョン確認
    try:
        import numpy as np
        print(f"現在のNumPy: {np.__version__}")
        if np.__version__.startswith("2."):
            print("❌ NumPy 2.0が検出されました - ダウングレードします")
            need_fix = True
        else:
            print("✅ NumPy 1.x - 問題なし")
            need_fix = False
    except ImportError:
        print("NumPyがインストールされていません")
        need_fix = True
    
    if need_fix:
        print("🔄 NumPy 1.x にダウングレード中...")
        
        # NumPy 2.0をアンインストール
        run_command("pip uninstall -y numpy")
        
        # NumPy 1.x をインストール
        success = run_command("pip install 'numpy<2.0'")
        
        if success:
            print("✅ NumPy修正完了")
            try:
                import numpy as np
                print(f"新しいNumPy: {np.__version__}")
            except ImportError:
                print("❌ NumPyインポートに失敗")
        else:
            print("❌ NumPy修正に失敗")
    
    print("=" * 50)
    print("🎉 修正完了 - setup.pyを再実行してください")

if __name__ == "__main__":
    main()