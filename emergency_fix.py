#!/usr/bin/env python3
"""
緊急修正スクリプト - PyTorchとNumPyの互換性問題解決
"""

import subprocess
import sys

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
    print("🚨 緊急修正: PyTorch/NumPy互換性問題")
    print("=" * 50)
    
    # 1. 全部アンインストール
    print("🗑️ 問題のあるパッケージ削除中...")
    packages_to_remove = [
        "torch", "torchvision", "torchaudio", 
        "numpy", "diffusers", "transformers", "huggingface_hub"
    ]
    
    for package in packages_to_remove:
        run_command(f"pip uninstall -y {package}", f"{package} 削除")
    
    # 2. 互換性のあるバージョンを順番にインストール
    print("📦 互換バージョン再インストール中...")
    
    # NumPy 1.x
    run_command('pip install "numpy>=1.21.0,<2.0"', "NumPy 1.x")
    
    # PyTorch 2.1.0 (NumPy 1.xと互換)
    run_command(
        "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118",
        "PyTorch 2.1.0 CUDA"
    )
    
    # Hugging Face Hub
    run_command("pip install huggingface_hub==0.20.3", "Hugging Face Hub")
    
    # Diffusers & Transformers
    run_command("pip install diffusers==0.25.1", "Diffusers")
    run_command("pip install transformers==4.36.0", "Transformers")
    
    # その他
    other_packages = ["accelerate", "safetensors", "pillow", "matplotlib"]
    for package in other_packages:
        run_command(f"pip install {package}", package)
    
    # 3. 動作確認
    print("🔍 動作確認中...")
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"✅ CUDA: {torch.cuda.is_available()}")
        
        from diffusers import StableDiffusionPipeline
        print("✅ Diffusers インポート成功")
        
        print("=" * 50)
        print("🎉 緊急修正完了！")
        print("📝 次のステップ:")
        print("   1. python quick_start.py でテスト")
        print("   2. または手動でパイプライン作成")
        
    except Exception as e:
        print(f"❌ 動作確認エラー: {e}")
        print("💡 Google Colabのランタイムを再起動してください")

if __name__ == "__main__":
    main()