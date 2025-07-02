#!/usr/bin/env python3
"""
setup.py実行後の問題修正スクリプト
"""

import subprocess

def run_command(command, description=""):
    print(f"🔄 {description}")
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} - 完了")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {description} - 失敗")
        return False

def main():
    print("🛠️ setup.py後の問題修正中...")
    print("=" * 50)
    
    # 1. NumPy問題修正
    print("🔧 NumPy修正...")
    run_command('pip install "numpy<2.0"', "NumPy 1.x インストール")
    
    # 2. Hugging Face Hub互換性修正
    print("🔧 Hugging Face Hub修正...")
    run_command("pip uninstall -y huggingface_hub", "古いHugging Face Hub削除")
    run_command("pip install huggingface_hub==0.20.3", "互換バージョンインストール")
    run_command("pip install diffusers==0.25.1 --force-reinstall", "Diffusers再インストール")
    
    # 3. 動作確認
    print("🔍 動作確認...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        from huggingface_hub import snapshot_download
        print("✅ Hugging Face Hub OK")
        
        from diffusers import StableDiffusionPipeline
        print("✅ Diffusers OK")
        
        print("=" * 50)
        print("🎉 修正完了！quick_start.pyを実行してください")
        
    except Exception as e:
        print(f"❌ 動作確認エラー: {e}")

if __name__ == "__main__":
    main()