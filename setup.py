#!/usr/bin/env python3
"""
SDXL Base 1.0 自動セットアップスクリプト
Google Colab環境でのStable Diffusion XL環境構築

使用方法:
    python setup.py

機能:
- 公式推奨依存関係の自動インストール
- SDXL Base 1.0モデルの準備
- Colab無料版用メモリ効率化設定
- 動作確認テスト
"""

import os
import sys
import subprocess
import time
import gc
from pathlib import Path

def print_header():
    """セットアップ開始メッセージ"""
    print("=" * 60)
    print("🚀 SDXL Base 1.0 自動セットアップ for Google Colab")
    print("=" * 60)
    print("📋 設定内容:")
    print("   - Stable Diffusion XL Base 1.0")
    print("   - Google Colab 無料版対応")
    print("   - 商用利用可能 (CreativeML Open RAIL++-M)")
    print("   - YouTube・Note用画像生成特化")
    print("=" * 60)

def check_environment():
    """環境チェック"""
    print("\n🔍 環境チェック...")
    
    # Python バージョン確認
    python_version = sys.version_info
    print(f"   Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8以上が必要です")
        return False
    
    # GPU確認
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA: {'利用可能' if cuda_available else '利用不可'}")
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU: {gpu_name}")
            print(f"   VRAM: {gpu_memory:.1f}GB")
            
            if gpu_memory < 8:
                print("⚠️  VRAM 8GB未満：メモリ効率化を強化します")
        else:
            print("❌ CUDA対応GPUが必要です")
            return False
            
    except ImportError:
        print("   PyTorch: 未インストール（後でインストール）")
    
    print("✅ 環境チェック完了")
    return True

def install_dependencies():
    """公式推奨依存関係のインストール"""
    print("\n📦 依存関係のインストール...")
    
    # 基本パッケージリスト（公式推奨順）
    packages = [
        # Core packages (公式推奨)
        ("diffusers>=0.19.0", "Diffusers (SDXL対応版)"),
        ("transformers", "Transformers"),
        ("safetensors", "SafeTensors"),
        ("accelerate", "Accelerate"),
        ("invisible_watermark", "Invisible Watermark"),
        
        # PyTorch (CUDA対応)
        ("torch torchvision --index-url https://download.pytorch.org/whl/cu118", "PyTorch (CUDA 11.8)"),
        
        # 画像処理・ユーティリティ
        ("pillow", "Pillow"),
        ("opencv-python", "OpenCV"),
        ("tqdm", "Progress Bar"),
        ("matplotlib", "Matplotlib"),
        
        # オプション：高速化
        ("xformers", "xFormers (オプション)"),
    ]
    
    successful_installs = []
    failed_installs = []
    
    for package_spec, description in packages:
        print(f"\n   📥 {description} をインストール中...")
        
        try:
            # xformersは失敗してもスキップ
            if "xformers" in package_spec:
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install"] + package_spec.split(),
                        capture_output=True, text=True, timeout=300
                    )
                except subprocess.TimeoutExpired:
                    print(f"   ⏰ タイムアウト: {description} (スキップ)")
                    continue
            else:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install"] + package_spec.split(),
                    capture_output=True, text=True, timeout=600
                )
            
            if result.returncode == 0:
                print(f"   ✅ {description} インストール成功")
                successful_installs.append(description)
            else:
                print(f"   ❌ {description} インストール失敗")
                if "xformers" not in package_spec:  # xformers以外は重要
                    print(f"      エラー: {result.stderr[:200]}")
                failed_installs.append(description)
                
        except Exception as e:
            print(f"   ❌ {description} インストール失敗: {str(e)[:100]}")
            failed_installs.append(description)
    
    print(f"\n📊 インストール結果:")
    print(f"   ✅ 成功: {len(successful_installs)}")
    print(f"   ❌ 失敗: {len(failed_installs)}")
    
    if failed_installs:
        print(f"   失敗したパッケージ: {', '.join(failed_installs)}")
    
    return len(failed_installs) == 0 or all("xformers" in pkg for pkg in failed_installs)

def setup_directories():
    """必要なディレクトリ作成"""
    print("\n📁 ディレクトリ構造を作成中...")
    
    directories = [
        "generated_images",
        "models_cache", 
        "logs",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   📂 {directory}/")
    
    print("✅ ディレクトリ作成完了")

def setup_sdxl_pipeline():
    """SDXL パイプラインのセットアップと初期化"""
    print("\n🎨 SDXL パイプラインを初期化中...")
    
    try:
        # Import all required libraries
        import torch
        from diffusers import DiffusionPipeline
        print("   📚 ライブラリ読み込み完了")
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("   🔄 SDXL Base 1.0 モデル読み込み中...")
        print("      (初回は数分かかる場合があります)")
        
        # Setup pipeline with official recommendations
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,      # Memory efficiency
            use_safetensors=True,           # Security
            variant="fp16"                  # Lightweight model
        )
        print("   ✅ モデル読み込み完了")
        
        # Apply Colab optimizations
        print("   ⚡ Google Colab 最適化を適用中...")
        
        # Essential optimizations for Colab free tier
        pipe.enable_model_cpu_offload()   # Official recommendation for low VRAM
        pipe.vae.enable_tiling()          # For high resolution generation
        
        # Optional optimizations (fail gracefully)
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("   ✅ xFormers メモリ最適化: 有効")
        except:
            print("   ⚠️  xFormers メモリ最適化: 無効 (スキップ)")
        
        try:
            # torch.compile for 20-30% speedup (if supported)
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            print("   ✅ torch.compile 高速化: 有効")
        except:
            print("   ⚠️  torch.compile 高速化: 無効 (サポート外)")
        
        print("✅ SDXL パイプライン準備完了")
        return pipe
        
    except Exception as e:
        print(f"❌ SDXL パイプライン初期化失敗: {e}")
        return None

def run_test_generation(pipe):
    """テスト画像生成"""
    print("\n🧪 テスト画像を生成中...")
    
    try:
        # Simple test prompt
        test_prompt = "A beautiful landscape with mountains and a lake, digital art, high quality"
        
        print(f"   📝 プロンプト: {test_prompt}")
        print("   ⏱️  生成中... (1-2分お待ちください)")
        
        start_time = time.time()
        
        # Generate test image
        images = pipe(
            prompt=test_prompt,
            height=1024,
            width=1024,
            num_inference_steps=20,  # Fast test
            guidance_scale=7.5,
            num_images_per_prompt=1
        ).images
        
        generation_time = time.time() - start_time
        
        # Save test image
        test_image = images[0]
        test_path = "generated_images/setup_test.png"
        test_image.save(test_path)
        
        print(f"   ✅ テスト画像生成成功")
        print(f"   📸 保存先: {test_path}")
        print(f"   ⏱️  生成時間: {generation_time:.2f}秒")
        print(f"   📏 解像度: {test_image.size}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ テスト画像生成失敗: {e}")
        return False

def cleanup_and_optimize():
    """メモリクリーンアップと最適化"""
    print("\n🧹 メモリクリーンアップ中...")
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # Memory status
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"   📊 GPU メモリ - 使用中: {allocated:.2f}GB, 予約済み: {reserved:.2f}GB")
        
        gc.collect()
        print("   ✅ メモリクリーンアップ完了")
        
    except Exception as e:
        print(f"   ⚠️  クリーンアップ警告: {e}")

def save_setup_info():
    """セットアップ情報の保存"""
    print("\n💾 セットアップ情報を保存中...")
    
    try:
        import torch
        import diffusers
        import transformers
        from datetime import datetime
        
        setup_info = {
            "setup_date": datetime.now().isoformat(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "torch_version": torch.__version__,
            "diffusers_version": diffusers.__version__,
            "transformers_version": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            setup_info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "cuda_version": torch.version.cuda,
            })
        
        # Save to file
        import json
        with open("setup_info.json", "w", encoding="utf-8") as f:
            json.dump(setup_info, f, indent=2, ensure_ascii=False)
        
        print("   ✅ セットアップ情報保存完了: setup_info.json")
        
    except Exception as e:
        print(f"   ⚠️  情報保存警告: {e}")

def print_completion_message():
    """完了メッセージ"""
    print("\n" + "=" * 60)
    print("🎉 SDXL セットアップ完了！")
    print("=" * 60)
    print("📋 次のステップ:")
    print("   1. python sdxl_test.py      # より詳細なテスト")
    print("   2. 生成された画像を確認      # generated_images/setup_test.png")
    print("   3. ドキュメントを参照        # docs/README.md")
    print("\n📚 主要コマンド:")
    print("   - YouTube サムネイル生成")
    print("   - Note 記事用画像生成") 
    print("   - アイコン・ロゴ作成")
    print("\n🔧 問題が発生した場合:")
    print("   python scripts/sdxl_fix.py  # 修正スクリプト実行")
    print("=" * 60)
    print("✨ 高品質な画像生成をお楽しみください！")
    print("=" * 60)

def main():
    """メイン処理"""
    start_time = time.time()
    
    try:
        # セットアップ手順
        print_header()
        
        if not check_environment():
            print("❌ 環境チェックに失敗しました")
            return False
        
        if not install_dependencies():
            print("❌ 依存関係のインストールに失敗しました")
            return False
        
        setup_directories()
        
        pipe = setup_sdxl_pipeline()
        if pipe is None:
            print("❌ SDXL パイプラインの初期化に失敗しました")
            return False
        
        if not run_test_generation(pipe):
            print("❌ テスト画像生成に失敗しました")
            return False
        
        cleanup_and_optimize()
        save_setup_info()
        
        total_time = time.time() - start_time
        print(f"\n⏱️  総セットアップ時間: {total_time/60:.1f}分")
        
        print_completion_message()
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️  セットアップが中断されました")
        return False
    except Exception as e:
        print(f"\n💥 予期しないエラー: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)