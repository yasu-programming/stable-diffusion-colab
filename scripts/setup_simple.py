#!/usr/bin/env python3
"""
Google Colab用 Stable Diffusion 簡易セットアップスクリプト
バージョン互換性問題の解決版
"""

import subprocess
import sys
import os

def run_command(command, description=""):
    """コマンドを実行"""
    if description:
        print(f"🔄 {description}")
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} - 完了")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ エラー: {description}")
        return False

def main():
    """簡易セットアップ"""
    print("🚀 Stable Diffusion 簡易セットアップ開始")
    print("=" * 50)
    
    # 1. PyTorchを先にインストール
    print("📦 PyTorch インストール中...")
    torch_success = run_command(
        "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118",
        "PyTorch (CUDA 11.8)"
    )
    
    if not torch_success:
        print("⚠️ CUDA版PyTorchに失敗、CPU版を試行...")
        run_command("pip install torch torchvision torchaudio", "PyTorch (CPU版)")
    
    # 2. Hugging Face Hubを先にアップグレード
    print("📦 Hugging Face Hub アップグレード中...")
    run_command("pip install --upgrade huggingface_hub", "Hugging Face Hub")
    
    # 3. Diffusersとその他
    print("📦 Diffusers インストール中...")
    packages = [
        "diffusers==0.25.1",
        "transformers==4.36.0", 
        "accelerate==0.25.0",
        "safetensors==0.4.0",
        "huggingface_hub>=0.20.0",
        "pillow",
        "numpy",
        "matplotlib"
    ]
    
    for package in packages:
        run_command(f"pip install {package}", package)
    
    # 3. ディレクトリ作成
    print("📁 ディレクトリ作成中...")
    directories = ["generated_images", "models_cache", "temp"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ {directory}/ 作成")
    
    # 4. 動作確認
    print("🔍 動作確認中...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"✅ CUDA利用可能: {torch.cuda.is_available()}")
        
        from diffusers import StableDiffusionPipeline
        print("✅ Diffusers インポート成功")
        
    except Exception as e:
        print(f"❌ 動作確認エラー: {e}")
    
    print("=" * 50)
    print("🎉 簡易セットアップ完了！")
    print("📝 次のステップ:")
    print("   手動でモデルをテストしてください:")
    print("   from diffusers import StableDiffusionPipeline")
    print("   pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')")

if __name__ == "__main__":
    main()