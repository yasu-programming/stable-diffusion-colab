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
    
    # 問題のあるパッケージを完全削除
    print("🗑️ 既存パッケージ削除中...")
    packages_to_remove = ["torch", "torchvision", "torchaudio", "numpy", "diffusers", "transformers", "huggingface_hub", "xformers"]
    for package in packages_to_remove:
        run_command(f"pip uninstall -y {package}", f"{package} 削除")
    
    # 現代的で確実に動作するバージョンでインストール
    print("🔧 NumPy 1.26.4 インストール...")
    run_command("pip install numpy==1.26.4", "NumPy 1.26.4固定")
    
    # PyTorch 2.2.2（NumPy 1.26.4と互換、現代的）
    torch_install = run_command(
        "pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118",
        "PyTorch 2.2.2 インストール"
    )
    
    if not torch_install:
        print("⚠️ CUDA版PyTorchに失敗、CPU版を試行中...")
        run_command("pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2", "PyTorch 2.2.2 (CPU版)")
    
    # Hugging Face Hub 現代版
    run_command("pip install huggingface_hub==0.24.6", "Hugging Face Hub 0.24.6")
    
    # その他のライブラリ（現代的で確実に動作するバージョン）
    packages = [
        "diffusers==0.30.0",  # 最新版（NumPy 1.26.4と互換）
        "transformers==4.44.0",  # 最新版
        "accelerate==0.34.0",  # 最新版
        "safetensors==0.4.4",   # 最新版
        "pillow>=9.0.0",
        "matplotlib>=3.5.0",
        "ipywidgets>=8.0.0"
    ]
    
    for package in packages:
        success = run_command(f"pip install {package}", f"{package} インストール")
        if not success:
            print(f"⚠️  {package} のインストールに失敗しましたが続行します")
    
    # xformers 現代版（オプション）
    print("🔧 xformers 現代版 インストール中...")
    xformers_success = run_command("pip install xformers==0.0.27", "xformers 0.0.27")
    if not xformers_success:
        print("⚠️ xformersインストールに失敗 - メモリ効率化なしで続行")

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
        
        # 現代的メモリ最適化テスト
        if torch.cuda.is_available():
            try:
                pipe.enable_memory_efficient_attention()
                print("✅ メモリ効率化テスト成功")
            except Exception as e:
                print(f"⚠️ メモリ効率化テスト失敗: {str(e)[:50]}")
        
        # メモリ解放
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✅ GPUメモリクリア完了")
        
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
        
        # 現代的メモリ最適化（エラーハンドリング付き）
        try:
            pipe.enable_memory_efficient_attention()
            print("✅ メモリ効率化 ON")
        except Exception as e:
            print(f"⚠️ メモリ効率化スキップ: {str(e)[:50]}")
        
        try:
            pipe.enable_model_cpu_offload()
            print("✅ CPU Offload ON")
        except Exception as e:
            print(f"⚠️ CPU Offloadスキップ: {str(e)[:50]}")
    
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