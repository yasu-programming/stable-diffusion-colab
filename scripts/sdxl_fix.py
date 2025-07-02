#!/usr/bin/env python3
"""
SDXL トラブルシューティング・修正スクリプト
Google Colab でのSDXL関連問題を自動修正

使用方法:
    python scripts/sdxl_fix.py

機能:
- 一般的な問題の自動診断
- メモリ関連問題の修正
- 依存関係の修正
- パフォーマンス最適化
- 段階的修復手順
"""

import os
import sys
import subprocess
import time
import gc
from pathlib import Path

def print_header():
    """修正スクリプト開始メッセージ"""
    print("=" * 60)
    print("🔧 SDXL トラブルシューティング & 修正スクリプト")
    print("=" * 60)
    print("🎯 対応可能な問題:")
    print("   - CUDA Out of Memory エラー")
    print("   - モデル読み込みエラー")
    print("   - 依存関係の問題")
    print("   - パフォーマンス低下")
    print("   - 生成失敗")
    print("=" * 60)

def diagnose_system():
    """システム診断"""
    print("\n🔍 システム診断中...")
    
    issues = []
    
    # Python・GPU確認
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated = torch.cuda.memory_allocated() / 1e9
            usage_percent = (allocated / gpu_memory) * 100
            
            print(f"   ✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            print(f"   📊 VRAM使用: {allocated:.2f}GB / {gpu_memory:.1f}GB ({usage_percent:.1f}%)")
            
            if gpu_memory < 8:
                issues.append("low_vram")
                print("   ⚠️  VRAM不足: 8GB未満のGPU")
            
            if usage_percent > 80:
                issues.append("high_memory_usage")
                print("   ⚠️  メモリ使用量高: 80%以上")
                
        else:
            issues.append("no_cuda")
            print("   ❌ CUDA: 利用不可")
            
    except ImportError:
        issues.append("no_pytorch")
        print("   ❌ PyTorch: 未インストール")
    
    # 依存関係確認
    required_packages = {
        'diffusers': '0.19.0',
        'transformers': None,
        'safetensors': None,
        'accelerate': None,
        'invisible_watermark': None
    }
    
    for package, min_version in required_packages.items():
        try:
            pkg = __import__(package)
            version = getattr(pkg, '__version__', 'unknown')
            print(f"   ✅ {package}: {version}")
            
            if min_version and version < min_version:
                issues.append(f"old_{package}")
                print(f"   ⚠️  {package}: バージョンが古い（要求: {min_version}以上）")
                
        except ImportError:
            issues.append(f"missing_{package}")
            print(f"   ❌ {package}: 未インストール")
    
    return issues

def fix_memory_issues():
    """メモリ関連問題の修正"""
    print("\n🧹 メモリ問題修正中...")
    
    try:
        import torch
        
        # 積極的なメモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("   ✅ CUDA メモリクリア完了")
        
        # Python ガベージコレクション
        gc.collect()
        print("   ✅ Python メモリクリア完了")
        
        # メモリ使用量確認
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   📊 クリア後VRAM: {allocated:.2f}GB / {total:.1f}GB")
        
        return True
        
    except Exception as e:
        print(f"   ❌ メモリクリア失敗: {e}")
        return False

def fix_dependencies():
    """依存関係の修正"""
    print("\n📦 依存関係修正中...")
    
    # 重要なパッケージの再インストール
    critical_packages = [
        "diffusers>=0.19.0",
        "transformers", 
        "safetensors",
        "accelerate"
    ]
    
    success = True
    
    for package in critical_packages:
        print(f"   🔄 {package} 修正中...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", package],
                capture_output=True, text=True, timeout=300
            )
            
            if result.returncode == 0:
                print(f"   ✅ {package} 修正完了")
            else:
                print(f"   ❌ {package} 修正失敗")
                success = False
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ {package} タイムアウト")
            success = False
        except Exception as e:
            print(f"   ❌ {package} エラー: {e}")
            success = False
    
    return success

def fix_model_loading():
    """モデル読み込み問題の修正"""
    print("\n🎨 モデル読み込み修正中...")
    
    try:
        # Hugging Face キャッシュクリア
        cache_dir = Path.home() / ".cache" / "huggingface"
        if cache_dir.exists():
            print("   🗑️  Hugging Face キャッシュクリア中...")
            # キャッシュクリアは慎重に（必要な部分のみ）
            transformers_cache = cache_dir / "transformers"
            if transformers_cache.exists():
                import shutil
                shutil.rmtree(transformers_cache)
                print("   ✅ Transformers キャッシュクリア完了")
        
        # モデル読み込みテスト
        print("   🧪 モデル読み込みテスト中...")
        
        import torch
        from diffusers import DiffusionPipeline
        
        # メモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # より保守的な設定でテスト
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            low_cpu_mem_usage=True
        )
        
        # 最小限の最適化
        pipe.enable_model_cpu_offload()
        
        print("   ✅ モデル読み込みテスト成功")
        
        # テスト後のクリーンアップ
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"   ❌ モデル読み込みテスト失敗: {e}")
        return False

def optimize_performance():
    """パフォーマンス最適化"""
    print("\n⚡ パフォーマンス最適化中...")
    
    optimizations = []
    
    try:
        import torch
        from diffusers import DiffusionPipeline
        
        # メモリ効率化のテスト
        print("   🧪 最適化機能テスト中...")
        
        # 一時的なパイプライン作成
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        
        # 基本最適化
        pipe.enable_model_cpu_offload()
        optimizations.append("CPU オフロード")
        
        pipe.vae.enable_tiling()
        optimizations.append("VAE タイリング")
        
        # オプション最適化テスト
        try:
            pipe.enable_xformers_memory_efficient_attention()
            optimizations.append("xFormers")
        except:
            print("   ⚠️  xFormers: 利用不可")
        
        try:
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
            optimizations.append("torch.compile")
        except:
            print("   ⚠️  torch.compile: 利用不可")
        
        # クリーンアップ
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"   ✅ 利用可能な最適化: {', '.join(optimizations)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 最適化テスト失敗: {e}")
        return False

def create_optimized_config():
    """最適化された設定ファイルの作成"""
    print("\n📝 最適化設定ファイル作成中...")
    
    try:
        config = {
            "model_settings": {
                "torch_dtype": "float16",
                "use_safetensors": True,
                "variant": "fp16",
                "low_cpu_mem_usage": True
            },
            "optimization_settings": {
                "enable_model_cpu_offload": True,
                "enable_vae_tiling": True,
                "enable_xformers": "auto",
                "enable_torch_compile": "auto"
            },
            "generation_settings": {
                "youtube_thumbnail": {
                    "height": 576,
                    "width": 1024,
                    "num_inference_steps": 25,
                    "guidance_scale": 8.0
                },
                "blog_image": {
                    "height": 536,
                    "width": 1024,
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5
                },
                "icon": {
                    "height": 1024,
                    "width": 1024,
                    "num_inference_steps": 35,
                    "guidance_scale": 9.0
                }
            },
            "memory_settings": {
                "max_batch_size": 1,
                "cleanup_interval": 5,
                "enable_attention_slicing": True
            }
        }
        
        import json
        with open("sdxl_optimized_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print("   ✅ 最適化設定保存: sdxl_optimized_config.json")
        return True
        
    except Exception as e:
        print(f"   ❌ 設定ファイル作成失敗: {e}")
        return False

def run_recovery_test():
    """修復テスト"""
    print("\n🧪 修復テスト実行中...")
    
    try:
        import torch
        from diffusers import DiffusionPipeline
        
        # メモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("   🔄 SDXL パイプライン初期化...")
        
        # 修復された設定でパイプライン作成
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            low_cpu_mem_usage=True
        )
        
        # 修復された最適化を適用
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        
        print("   🎨 テスト画像生成中...")
        
        # 簡単なテスト生成
        test_image = pipe(
            prompt="A simple test image, digital art",
            height=512,  # 小さめの解像度でテスト
            width=512,
            num_inference_steps=15,  # 短時間でテスト
            guidance_scale=7.0,
            num_images_per_prompt=1
        ).images[0]
        
        # 保存
        os.makedirs("generated_images", exist_ok=True)
        test_path = "generated_images/recovery_test.png"
        test_image.save(test_path)
        
        print(f"   ✅ テスト成功: {test_path}")
        
        # クリーンアップ
        del pipe, test_image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"   ❌ 修復テスト失敗: {e}")
        return False

def print_recommendations(issues):
    """推奨事項の表示"""
    print("\n💡 推奨事項とベストプラクティス:")
    
    if "low_vram" in issues:
        print("   🔸 VRAM不足対策:")
        print("      - 解像度を下げる (768x768 以下)")
        print("      - バッチサイズを1に制限")
        print("      - enable_sequential_cpu_offload() を使用")
    
    if "high_memory_usage" in issues:
        print("   🔸 メモリ使用量削減:")
        print("      - 定期的に torch.cuda.empty_cache() 実行")
        print("      - 不要な変数を del で削除")
        print("      - gc.collect() でガベージコレクション")
    
    if any("missing_" in issue for issue in issues):
        print("   🔸 依存関係修正:")
        print("      - python setup.py を再実行")
        print("      - pip install --upgrade を個別実行")
    
    print("\n   🔸 一般的なベストプラクティス:")
    print("      - 生成前後でメモリクリア")
    print("      - 適切な解像度設定")
    print("      - 段階的なステップ数調整")
    print("      - 定期的なキャッシュクリア")

def main():
    """メイン修復処理"""
    start_time = time.time()
    
    try:
        print_header()
        
        # システム診断
        issues = diagnose_system()
        
        if not issues:
            print("\n✅ 問題は検出されませんでした")
            return True
        
        print(f"\n🔧 検出された問題: {len(issues)}件")
        for issue in issues:
            print(f"   - {issue}")
        
        # 修復処理実行
        repairs_attempted = 0
        repairs_successful = 0
        
        # メモリ問題修正
        if any("memory" in issue for issue in issues):
            repairs_attempted += 1
            if fix_memory_issues():
                repairs_successful += 1
        
        # 依存関係修正
        if any("missing_" in issue or "old_" in issue for issue in issues):
            repairs_attempted += 1
            if fix_dependencies():
                repairs_successful += 1
        
        # モデル読み込み修正
        repairs_attempted += 1
        if fix_model_loading():
            repairs_successful += 1
        
        # パフォーマンス最適化
        repairs_attempted += 1
        if optimize_performance():
            repairs_successful += 1
        
        # 最適化設定作成
        create_optimized_config()
        
        # 修復テスト
        print("\n🏁 修復完了テスト...")
        test_success = run_recovery_test()
        
        # 結果表示
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("📋 修復結果サマリー")
        print("=" * 60)
        print(f"🔧 修復試行: {repairs_attempted}")
        print(f"✅ 修復成功: {repairs_successful}")
        print(f"🧪 最終テスト: {'成功' if test_success else '失敗'}")
        print(f"⏱️  修復時間: {total_time:.1f}秒")
        
        if test_success:
            print("\n🎉 修復が完了しました！")
            print("   通常の画像生成が可能になりました。")
        else:
            print("\n⚠️  一部の問題が解決できませんでした。")
            print("   手動での設定調整が必要な場合があります。")
        
        # 推奨事項表示
        print_recommendations(issues)
        
        print("=" * 60)
        
        return test_success
        
    except KeyboardInterrupt:
        print("\n⏹️  修復が中断されました")
        return False
    except Exception as e:
        print(f"\n💥 修復中にエラー: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)