#!/usr/bin/env python3
"""
SDXL メモリ最適化スクリプト
Google Colab 環境でのメモリ効率を最大化

使用方法:
    python scripts/memory_optimizer.py

機能:
- メモリ使用量の詳細分析
- 積極的なメモリクリーンアップ
- 最適化設定の自動調整
- メモリ効率的な画像生成設定
"""

import gc
import os
import sys
import time
import psutil
from pathlib import Path

def print_header():
    """メモリ最適化開始メッセージ"""
    print("=" * 60)
    print("🧠 SDXL メモリ最適化スクリプト")
    print("=" * 60)
    print("🎯 最適化内容:")
    print("   - GPU・RAMメモリの詳細分析")
    print("   - 積極的なメモリクリーンアップ")
    print("   - 最適化設定の自動調整")
    print("   - メモリ効率的な生成パラメータ")
    print("=" * 60)

def analyze_memory_usage():
    """詳細メモリ分析"""
    print("\n📊 メモリ使用量分析中...")
    
    memory_info = {}
    
    # RAM使用量
    ram = psutil.virtual_memory()
    memory_info['ram'] = {
        'total_gb': ram.total / 1e9,
        'used_gb': ram.used / 1e9,
        'available_gb': ram.available / 1e9,
        'usage_percent': ram.percent
    }
    
    print(f"   💾 RAM:")
    print(f"      総容量: {memory_info['ram']['total_gb']:.1f}GB")
    print(f"      使用中: {memory_info['ram']['used_gb']:.1f}GB")
    print(f"      利用可能: {memory_info['ram']['available_gb']:.1f}GB")
    print(f"      使用率: {memory_info['ram']['usage_percent']:.1f}%")
    
    # GPU メモリ
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            
            memory_info['gpu'] = {
                'total_gb': total / 1e9,
                'allocated_gb': allocated / 1e9,
                'reserved_gb': reserved / 1e9,
                'free_gb': (total - reserved) / 1e9,
                'usage_percent': (allocated / total) * 100
            }
            
            print(f"   🎮 GPU VRAM:")
            print(f"      総容量: {memory_info['gpu']['total_gb']:.1f}GB")
            print(f"      使用中: {memory_info['gpu']['allocated_gb']:.2f}GB")
            print(f"      予約済み: {memory_info['gpu']['reserved_gb']:.2f}GB")
            print(f"      空き: {memory_info['gpu']['free_gb']:.2f}GB")
            print(f"      使用率: {memory_info['gpu']['usage_percent']:.1f}%")
        else:
            print("   ❌ GPU VRAM: CUDA利用不可")
            memory_info['gpu'] = None
    except ImportError:
        print("   ❌ GPU VRAM: PyTorch未インストール")
        memory_info['gpu'] = None
    
    return memory_info

def aggressive_memory_cleanup():
    """積極的メモリクリーンアップ"""
    print("\n🧹 積極的メモリクリーンアップ実行中...")
    
    cleanup_results = {}
    
    # PyTorch GPU メモリクリア
    try:
        import torch
        if torch.cuda.is_available():
            before_allocated = torch.cuda.memory_allocated() / 1e9
            before_reserved = torch.cuda.memory_reserved() / 1e9
            
            # 全てのキャッシュをクリア
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            # 複数回実行して確実にクリア
            for _ in range(3):
                torch.cuda.empty_cache()
                time.sleep(0.1)
            
            after_allocated = torch.cuda.memory_allocated() / 1e9
            after_reserved = torch.cuda.memory_reserved() / 1e9
            
            cleanup_results['gpu'] = {
                'before_allocated': before_allocated,
                'after_allocated': after_allocated,
                'before_reserved': before_reserved,
                'after_reserved': after_reserved,
                'freed_allocated': before_allocated - after_allocated,
                'freed_reserved': before_reserved - after_reserved
            }
            
            print(f"   🎮 GPU メモリクリア:")
            print(f"      解放 (使用中): {cleanup_results['gpu']['freed_allocated']:.2f}GB")
            print(f"      解放 (予約済み): {cleanup_results['gpu']['freed_reserved']:.2f}GB")
            
    except Exception as e:
        print(f"   ❌ GPU メモリクリア失敗: {e}")
    
    # Python ガベージコレクション
    before_objects = len(gc.get_objects())
    collected = gc.collect()
    after_objects = len(gc.get_objects())
    
    cleanup_results['python'] = {
        'objects_before': before_objects,
        'objects_after': after_objects,
        'objects_collected': collected,
        'objects_reduced': before_objects - after_objects
    }
    
    print(f"   🐍 Python ガベージコレクション:")
    print(f"      回収されたオブジェクト: {cleanup_results['python']['objects_collected']}")
    print(f"      削減されたオブジェクト: {cleanup_results['python']['objects_reduced']}")
    
    # システムキャッシュの確認（参考情報）
    try:
        # /proc/meminfoが利用可能な場合（Linux）
        if os.path.exists('/proc/meminfo'):
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                for line in meminfo.split('\n'):
                    if 'Cached:' in line:
                        cached_kb = int(line.split()[1])
                        cached_gb = cached_kb / 1e6
                        print(f"   💾 システムキャッシュ: {cached_gb:.2f}GB")
                        break
    except:
        pass
    
    return cleanup_results

def optimize_pipeline_settings():
    """パイプライン最適化設定"""
    print("\n⚙️ パイプライン最適化設定生成中...")
    
    try:
        # メモリ状況に基づく設定
        memory_info = analyze_memory_usage()
        
        # GPU VRAM に基づく設定調整
        if memory_info['gpu']:
            vram_gb = memory_info['gpu']['total_gb']
            
            if vram_gb >= 12:
                tier = "high"
                print("   🚀 高性能設定 (12GB以上)")
            elif vram_gb >= 8:
                tier = "medium"
                print("   ⚡ 標準設定 (8-12GB)")
            else:
                tier = "low"
                print("   🔋 省メモリ設定 (8GB未満)")
        else:
            tier = "low"
            print("   🔋 CPU設定 (CUDA無効)")
        
        # 設定定義
        settings = {
            "high": {
                "model_settings": {
                    "torch_dtype": "float16",
                    "use_safetensors": True,
                    "variant": "fp16"
                },
                "optimization": {
                    "enable_model_cpu_offload": False,
                    "enable_sequential_cpu_offload": False,
                    "enable_vae_tiling": True,
                    "enable_attention_slicing": False
                },
                "generation_limits": {
                    "max_height": 1536,
                    "max_width": 1536,
                    "max_batch_size": 2,
                    "max_inference_steps": 50
                }
            },
            "medium": {
                "model_settings": {
                    "torch_dtype": "float16",
                    "use_safetensors": True,
                    "variant": "fp16"
                },
                "optimization": {
                    "enable_model_cpu_offload": True,
                    "enable_sequential_cpu_offload": False,
                    "enable_vae_tiling": True,
                    "enable_attention_slicing": True
                },
                "generation_limits": {
                    "max_height": 1024,
                    "max_width": 1024,
                    "max_batch_size": 1,
                    "max_inference_steps": 40
                }
            },
            "low": {
                "model_settings": {
                    "torch_dtype": "float16",
                    "use_safetensors": True,
                    "variant": "fp16"
                },
                "optimization": {
                    "enable_model_cpu_offload": True,
                    "enable_sequential_cpu_offload": True,
                    "enable_vae_tiling": True,
                    "enable_attention_slicing": True
                },
                "generation_limits": {
                    "max_height": 768,
                    "max_width": 768,
                    "max_batch_size": 1,
                    "max_inference_steps": 30
                }
            }
        }
        
        selected_settings = settings[tier]
        
        # 用途別推奨設定
        use_case_settings = {
            "youtube_thumbnail": {
                "height": min(576, selected_settings["generation_limits"]["max_height"]),
                "width": min(1024, selected_settings["generation_limits"]["max_width"]),
                "num_inference_steps": min(25, selected_settings["generation_limits"]["max_inference_steps"]),
                "guidance_scale": 8.0
            },
            "blog_image": {
                "height": min(536, selected_settings["generation_limits"]["max_height"]),
                "width": min(1024, selected_settings["generation_limits"]["max_width"]),
                "num_inference_steps": min(30, selected_settings["generation_limits"]["max_inference_steps"]),
                "guidance_scale": 7.5
            },
            "icon": {
                "height": min(1024, selected_settings["generation_limits"]["max_height"]),
                "width": min(1024, selected_settings["generation_limits"]["max_width"]),
                "num_inference_steps": min(35, selected_settings["generation_limits"]["max_inference_steps"]),
                "guidance_scale": 9.0
            }
        }
        
        # 設定保存
        import json
        config = {
            "memory_tier": tier,
            "vram_gb": memory_info['gpu']['total_gb'] if memory_info['gpu'] else 0,
            "pipeline_settings": selected_settings,
            "use_case_settings": use_case_settings,
            "memory_management": {
                "cleanup_interval": 5,
                "enable_gc_collect": True,
                "enable_torch_cuda_empty_cache": True
            }
        }
        
        with open("memory_optimized_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ 最適化設定保存: memory_optimized_config.json")
        print(f"   📊 設定レベル: {tier}")
        
        return config
        
    except Exception as e:
        print(f"   ❌ 設定生成失敗: {e}")
        return None

def create_memory_monitor():
    """メモリ監視スクリプト作成"""
    print("\n📈 メモリ監視スクリプト作成中...")
    
    monitor_script = '''
import torch
import gc
import time

class MemoryMonitor:
    def __init__(self):
        self.peak_gpu_memory = 0
        self.peak_ram_usage = 0
        
    def check_memory(self, label=""):
        """メモリ使用量チェック"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            self.peak_gpu_memory = max(self.peak_gpu_memory, allocated)
            print(f"[{label}] GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved (peak: {self.peak_gpu_memory:.2f}GB)")
        
        import psutil
        ram_usage = psutil.virtual_memory().percent
        self.peak_ram_usage = max(self.peak_ram_usage, ram_usage)
        print(f"[{label}] RAM: {ram_usage:.1f}% (peak: {self.peak_ram_usage:.1f}%)")
    
    def cleanup_memory(self):
        """メモリクリーンアップ"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Memory cleanup completed")

# 使用例:
# monitor = MemoryMonitor()
# monitor.check_memory("Before generation")
# # ... 画像生成処理 ...
# monitor.check_memory("After generation")
# monitor.cleanup_memory()
'''
    
    try:
        with open("memory_monitor.py", "w", encoding="utf-8") as f:
            f.write(monitor_script)
        print("   ✅ メモリ監視スクリプト作成: memory_monitor.py")
        return True
    except Exception as e:
        print(f"   ❌ 監視スクリプト作成失敗: {e}")
        return False

def test_optimized_generation():
    """最適化された設定でのテスト生成"""
    print("\n🧪 最適化設定テスト中...")
    
    try:
        import torch
        from diffusers import DiffusionPipeline
        import json
        
        # 最適化設定読み込み
        if os.path.exists("memory_optimized_config.json"):
            with open("memory_optimized_config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            print("   ⚠️  最適化設定ファイルが見つかりません")
            return False
        
        # メモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("   🔄 最適化設定でパイプライン初期化...")
        
        # パイプライン作成
        pipe_settings = config["pipeline_settings"]["model_settings"]
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            **pipe_settings
        )
        
        # 最適化適用
        opt_settings = config["pipeline_settings"]["optimization"]
        
        if opt_settings["enable_model_cpu_offload"]:
            pipe.enable_model_cpu_offload()
            print("   ✅ CPU オフロード有効")
        
        if opt_settings["enable_sequential_cpu_offload"]:
            pipe.enable_sequential_cpu_offload()
            print("   ✅ シーケンシャル CPU オフロード有効")
        
        if opt_settings["enable_vae_tiling"]:
            pipe.vae.enable_tiling()
            print("   ✅ VAE タイリング有効")
        
        if opt_settings["enable_attention_slicing"]:
            pipe.enable_attention_slicing()
            print("   ✅ Attention スライシング有効")
        
        # メモリ監視開始
        if torch.cuda.is_available():
            before_memory = torch.cuda.memory_allocated() / 1e9
            print(f"   📊 生成前GPU使用量: {before_memory:.2f}GB")
        
        # テスト生成（最小限の設定）
        test_settings = config["use_case_settings"]["youtube_thumbnail"]
        test_settings["height"] = min(512, test_settings["height"])  # さらに小さく
        test_settings["width"] = min(512, test_settings["width"])
        test_settings["num_inference_steps"] = min(15, test_settings["num_inference_steps"])
        
        print("   🎨 テスト画像生成中...")
        start_time = time.time()
        
        test_image = pipe(
            prompt="Simple test image, digital art",
            **test_settings,
            num_images_per_prompt=1
        ).images[0]
        
        generation_time = time.time() - start_time
        
        # メモリ監視終了
        if torch.cuda.is_available():
            after_memory = torch.cuda.memory_allocated() / 1e9
            print(f"   📊 生成後GPU使用量: {after_memory:.2f}GB")
            print(f"   📊 GPU使用量増加: {after_memory - before_memory:.2f}GB")
        
        # 保存
        os.makedirs("generated_images", exist_ok=True)
        test_path = "generated_images/memory_optimized_test.png"
        test_image.save(test_path)
        
        print(f"   ✅ テスト成功!")
        print(f"   ⏱️  生成時間: {generation_time:.2f}秒")
        print(f"   📸 保存先: {test_path}")
        print(f"   📏 解像度: {test_image.size}")
        
        # クリーンアップ
        del pipe, test_image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"   ❌ テスト失敗: {e}")
        return False

def print_memory_tips():
    """メモリ最適化のコツ"""
    print("\n💡 メモリ最適化のコツ:")
    print("   🔸 生成前後の習慣:")
    print("      - torch.cuda.empty_cache() でGPUメモリクリア")
    print("      - gc.collect() でPythonメモリクリア") 
    print("      - 不要な変数は del で削除")
    
    print("\n   🔸 生成設定の調整:")
    print("      - 解像度を段階的に下げる")
    print("      - バッチサイズを1に制限")
    print("      - 推論ステップ数を削減")
    
    print("\n   🔸 パイプライン最適化:")
    print("      - enable_model_cpu_offload() 使用")
    print("      - enable_vae_tiling() で高解像度対応")
    print("      - enable_attention_slicing() でメモリ節約")
    
    print("\n   🔸 緊急時の対処:")
    print("      - Colabランタイムを再起動")
    print("      - より小さい解像度で再試行")
    print("      - enable_sequential_cpu_offload() 使用")

def main():
    """メイン最適化処理"""
    start_time = time.time()
    
    try:
        print_header()
        
        # メモリ分析
        memory_before = analyze_memory_usage()
        
        # 積極的クリーンアップ
        cleanup_results = aggressive_memory_cleanup()
        
        # 最適化後のメモリ状況
        print("\n📊 クリーンアップ後のメモリ状況:")
        memory_after = analyze_memory_usage()
        
        # 最適化設定生成
        config = optimize_pipeline_settings()
        
        # 監視スクリプト作成
        create_memory_monitor()
        
        # テスト実行
        test_success = test_optimized_generation()
        
        # 結果サマリー
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("📋 メモリ最適化結果")
        print("=" * 60)
        
        if memory_before['gpu'] and memory_after['gpu']:
            gpu_improvement = memory_before['gpu']['usage_percent'] - memory_after['gpu']['usage_percent']
            print(f"🎮 GPU メモリ改善: {gpu_improvement:.1f}%")
        
        ram_improvement = memory_before['ram']['usage_percent'] - memory_after['ram']['usage_percent']
        print(f"💾 RAM 改善: {ram_improvement:.1f}%")
        
        print(f"🧪 最適化テスト: {'成功' if test_success else '失敗'}")
        print(f"⏱️  最適化時間: {total_time:.1f}秒")
        
        if test_success:
            print("\n🎉 メモリ最適化が完了しました！")
            print("   memory_optimized_config.json を使用してください")
        else:
            print("\n⚠️  一部の最適化が完了しませんでした")
        
        # コツの表示
        print_memory_tips()
        
        print("=" * 60)
        
        return test_success
        
    except KeyboardInterrupt:
        print("\n⏹️  最適化が中断されました")
        return False
    except Exception as e:
        print(f"\n💥 最適化中にエラー: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)