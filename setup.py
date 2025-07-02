#!/usr/bin/env python3
"""
Google Colab用 Stable Diffusion セットアップスクリプト
使用方法: python setup.py
"""

import subprocess
import sys
import os
import torch
from pathlib import Path

def run_command(command, description=""):
    """コマンドを実行し、結果を表示"""
    if description:
        print(f"🔄 {description}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        if result.stdout:
            print(f"✅ {description} - 完了")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ エラー: {description}")
        print(f"コマンド: {command}")
        print(f"エラー詳細: {e.stderr}")
        return False

def check_gpu():
    """GPU利用可能性をチェック"""
    print("🔍 GPU確認中...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU利用可能: {gpu_name}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("❌ GPU利用不可 - CPUモードで動作します（非常に遅い）")
        return False

def install_dependencies():
    """必要なライブラリをインストール"""
    print("📦 ライブラリインストール中...")
    
    # PyTorchを先にインストール（バージョン固定）
    torch_install = run_command(
        "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118",
        "PyTorch インストール"
    )
    
    if not torch_install:
        print("⚠️ PyTorchインストールに失敗、デフォルト版を試行中...")
        run_command("pip install torch torchvision torchaudio", "PyTorch (デフォルト版)")
    
    # Hugging Face Hub を先にアップグレード
    run_command("pip install --upgrade huggingface_hub", "Hugging Face Hub アップグレード")
    
    # その他のライブラリ（バージョン互換性考慮）
    packages = [
        "diffusers==0.25.1",  # 安定版
        "transformers==4.36.0",  # 互換性確認済み
        "accelerate==0.25.0",
        "safetensors==0.4.0",
        "huggingface_hub>=0.20.0",  # 明示的にバージョン指定
        "pillow>=9.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "ipywidgets>=8.0.0"
    ]
    
    for package in packages:
        success = run_command(f"pip install {package}", f"{package} インストール")
        if not success:
            print(f"⚠️  {package} のインストールに失敗しましたが続行します")
    
    # xformersは最後に（オプション）
    print("🔧 xformers インストール中（メモリ効率化）...")
    run_command("pip install xformers --no-deps", "xformers インストール")

def setup_directories():
    """必要なディレクトリを作成"""
    print("📁 ディレクトリ設定中...")
    
    directories = [
        "generated_images",
        "models_cache", 
        "temp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ {directory}/ ディレクトリ作成")

def download_models():
    """推奨モデルをプリロード"""
    print("🤖 モデルダウンロード中...")
    
    try:
        # インポートテスト
        print("🔍 ライブラリインポート確認中...")
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        from diffusers import StableDiffusionPipeline
        print("✅ Diffusers インポート成功")
        
        # SD 1.5をキャッシュにダウンロード
        print("📥 Stable Diffusion v1.5 ダウンロード中...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            cache_dir="./models_cache",
            safety_checker=None,
            requires_safety_checker=False
        )
        print("✅ SD v1.5 ダウンロード完了")
        
        # メモリ解放
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        print("💡 再インストールを試してください: pip install --upgrade diffusers transformers")
        return False
    except Exception as e:
        print(f"❌ モデルダウンロードエラー: {e}")
        print("💡 手動でモデルを確認してください")
        return False

def create_quick_start():
    """クイックスタート用のサンプルコードを作成"""
    print("📝 クイックスタートファイル作成中...")
    
    quick_start_code = '''
# クイックスタート用サンプルコード
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt

def generate_image(prompt, negative_prompt="", width=512, height=512):
    """画像生成関数"""
    # パイプライン初期化（初回のみ時間がかかります）
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        cache_dir="./models_cache"
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
    
    # 画像生成
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=20,
        guidance_scale=7.5
    ).images[0]
    
    return image

# 使用例
if __name__ == "__main__":
    # YouTubeサムネイル例
    prompt = "modern tech office, programmer, clean design, professional lighting"
    negative_prompt = "blurry, low quality, messy"
    
    image = generate_image(prompt, negative_prompt, width=896, height=512)
    
    # 表示と保存
    plt.figure(figsize=(12, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Generated: {prompt[:50]}...")
    plt.show()
    
    image.save("generated_images/sample_thumbnail.png")
    print("画像を generated_images/sample_thumbnail.png に保存しました")
'''
    
    with open("quick_start.py", "w", encoding="utf-8") as f:
        f.write(quick_start_code)
    
    print("✅ quick_start.py 作成完了")

def main():
    """メイン実行関数"""
    print("🚀 Stable Diffusion Colab セットアップ開始")
    print("=" * 50)
    
    # GPU確認
    gpu_available = check_gpu()
    
    # ライブラリインストール
    install_dependencies()
    
    # ディレクトリ設定
    setup_directories()
    
    # モデルダウンロード
    if gpu_available:
        download_models()
    else:
        print("⚠️  GPU利用不可のためモデルダウンロードをスキップ")
    
    # クイックスタートファイル作成
    create_quick_start()
    
    print("=" * 50)
    print("🎉 セットアップ完了！")
    print("📝 次のステップ:")
    print("   1. python quick_start.py で画像生成テスト")
    print("   2. generated_images/ フォルダで生成画像を確認")
    print("   3. docs/ フォルダの使用例を参考にカスタマイズ")
    
    if gpu_available:
        print("⚡ GPU利用可能 - 高速生成できます")
    else:
        print("🐌 CPU利用 - 生成に時間がかかります")

if __name__ == "__main__":
    main()