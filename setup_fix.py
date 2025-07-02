#!/usr/bin/env python3
"""
Hugging Face Hub エラー修正用スクリプト
"""

import subprocess
import sys

def run_command(command, description=""):
    print(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} - 完了")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {description} - 失敗")
        return False

def main():
    print("🛠️ Hugging Face Hub エラー修正中...")
    print("=" * 50)
    
    # 1. 古いバージョンをアンインストール
    print("🗑️ 古いパッケージ削除中...")
    run_command("pip uninstall -y huggingface_hub diffusers transformers", "古いパッケージ削除")
    
    # 2. 最新のHugging Face Hubをインストール
    print("📦 最新Hugging Face Hubインストール...")
    run_command("pip install --upgrade huggingface_hub", "Hugging Face Hub")
    
    # 3. 互換性のあるDiffusersをインストール
    print("📦 Diffusers インストール...")
    run_command("pip install diffusers==0.25.1", "Diffusers")
    
    # 4. Transformersをインストール
    print("📦 Transformers インストール...")
    run_command("pip install transformers==4.36.0", "Transformers")
    
    # 5. その他必要なパッケージ
    packages = ["accelerate", "safetensors", "pillow", "numpy", "matplotlib"]
    for package in packages:
        run_command(f"pip install {package}", package)
    
    # 6. 動作確認
    print("🔍 動作確認中...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        from huggingface_hub import snapshot_download
        print("✅ Hugging Face Hub インポート成功")
        
        from diffusers import StableDiffusionPipeline
        print("✅ Diffusers インポート成功")
        
        print("=" * 50)
        print("🎉 修正完了！setup.pyを再実行してください")
        
    except Exception as e:
        print(f"❌ 動作確認エラー: {e}")
        print("💡 Google Colabのランタイムを再起動してから再試行してください")

if __name__ == "__main__":
    main()