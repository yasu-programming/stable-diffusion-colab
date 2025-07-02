#!/usr/bin/env python3
"""
SDXL Base 1.0 詳細テストスクリプト
セットアップ完了後の動作確認とパフォーマンステスト

使用方法:
    python sdxl_test.py

機能:
- SDXL パイプライン動作確認
- 各用途別画像生成テスト
- パフォーマンス測定
- メモリ使用量チェック
- 品質評価
"""

import os
import sys
import time
import json
import gc
from datetime import datetime
from pathlib import Path

def print_header():
    """テスト開始メッセージ"""
    print("=" * 60)
    print("🧪 SDXL Base 1.0 詳細テストスクリプト")
    print("=" * 60)
    print("📋 テスト内容:")
    print("   - 環境・パイプライン確認")
    print("   - YouTube サムネイル生成テスト")
    print("   - Note・ブログ画像生成テスト")
    print("   - アイコン・ロゴ生成テスト")
    print("   - パフォーマンス測定")
    print("   - メモリ使用量チェック")
    print("=" * 60)

def check_setup():
    """セットアップ確認"""
    print("\n🔍 セットアップ確認中...")
    
    setup_checks = {
        'torch': False,
        'diffusers': False,
        'transformers': False,
        'cuda': False,
        'sdxl_model': False
    }
    
    try:
        import torch
        setup_checks['torch'] = True
        print(f"   ✅ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            setup_checks['cuda'] = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   ✅ CUDA: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("   ❌ CUDA: 利用不可")
            
    except ImportError:
        print("   ❌ PyTorch: 未インストール")
    
    try:
        import diffusers
        setup_checks['diffusers'] = True
        print(f"   ✅ Diffusers: {diffusers.__version__}")
    except ImportError:
        print("   ❌ Diffusers: 未インストール")
    
    try:
        import transformers
        setup_checks['transformers'] = True
        print(f"   ✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("   ❌ Transformers: 未インストール")
    
    # 必要なディレクトリ確認
    required_dirs = ['generated_images', 'logs']
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"   ✅ Directory: {directory}/")
        else:
            Path(directory).mkdir(exist_ok=True)
            print(f"   📁 Created: {directory}/")
    
    # 基本的なセットアップが完了しているかチェック
    basic_setup_ok = all([
        setup_checks['torch'],
        setup_checks['diffusers'], 
        setup_checks['transformers'],
        setup_checks['cuda']
    ])
    
    if basic_setup_ok:
        print("✅ 基本セットアップ確認完了")
    else:
        print("❌ セットアップに問題があります。setup.py を再実行してください。")
        return False
    
    return True

def initialize_pipeline():
    """SDXL パイプライン初期化"""
    print("\n🎨 SDXL パイプライン初期化中...")
    
    try:
        import torch
        from diffusers import DiffusionPipeline
        
        # メモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("   🔄 モデル読み込み中...")
        start_time = time.time()
        
        # SDXL パイプライン初期化
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        
        load_time = time.time() - start_time
        print(f"   ✅ モデル読み込み完了 ({load_time:.2f}秒)")
        
        # 最適化適用
        print("   ⚡ 最適化適用中...")
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        
        # オプション最適化
        optimizations = []
        try:
            pipe.enable_xformers_memory_efficient_attention()
            optimizations.append("xFormers")
        except:
            pass
        
        try:
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            optimizations.append("torch.compile")
        except:
            pass
        
        if optimizations:
            print(f"   ✅ 最適化有効: {', '.join(optimizations)}")
        else:
            print("   ⚠️  追加最適化なし")
        
        print("✅ パイプライン初期化完了")
        return pipe
        
    except Exception as e:
        print(f"❌ パイプライン初期化失敗: {e}")
        return None

def test_youtube_generation(pipe):
    """YouTube サムネイル生成テスト"""
    print("\n🎬 YouTube サムネイル生成テスト...")
    
    test_prompts = [
        {
            "name": "gaming",
            "prompt": "Excited gamer with headphones playing video game, colorful RGB background, high energy",
            "style": "gaming"
        },
        {
            "name": "tech_tutorial", 
            "prompt": "Professional programmer at clean desk with multiple monitors, coding tutorial",
            "style": "educational"
        }
    ]
    
    results = []
    
    for test in test_prompts:
        print(f"\n   📝 テスト: {test['name']}")
        print(f"      プロンプト: {test['prompt'][:50]}...")
        
        try:
            start_time = time.time()
            
            images = pipe(
                prompt=f"{test['prompt']}, YouTube thumbnail style, professional photography, ultra detailed",
                negative_prompt="blurry, low quality, boring, text overlay",
                height=576,  # 16:9
                width=1024,
                num_inference_steps=25,
                guidance_scale=8.0,
                num_images_per_prompt=1
            ).images
            
            generation_time = time.time() - start_time
            
            # 保存
            filename = f"generated_images/test_youtube_{test['name']}.png"
            images[0].save(filename)
            
            print(f"      ✅ 成功 ({generation_time:.2f}秒) - {filename}")
            
            results.append({
                'type': 'youtube',
                'name': test['name'],
                'success': True,
                'time': generation_time,
                'file': filename,
                'size': images[0].size
            })
            
        except Exception as e:
            print(f"      ❌ 失敗: {e}")
            results.append({
                'type': 'youtube',
                'name': test['name'],
                'success': False,
                'error': str(e)
            })
    
    return results

def test_blog_generation(pipe):
    """ブログ画像生成テスト"""
    print("\n📝 ブログ画像生成テスト...")
    
    test_prompts = [
        {
            "name": "tech_workspace",
            "prompt": "Modern laptop on clean desk with coffee cup and notebook, minimalist workspace",
            "category": "tech"
        },
        {
            "name": "lifestyle_cozy",
            "prompt": "Cozy living room with warm lighting, books and plants, comfortable atmosphere",
            "category": "lifestyle"
        }
    ]
    
    results = []
    
    for test in test_prompts:
        print(f"\n   📝 テスト: {test['name']}")
        print(f"      プロンプト: {test['prompt'][:50]}...")
        
        try:
            start_time = time.time()
            
            images = pipe(
                prompt=f"{test['prompt']}, blog header image, professional photography, high quality",
                negative_prompt="cluttered, messy, unprofessional, low quality",
                height=536,  # 1.91:1
                width=1024,
                num_inference_steps=30,
                guidance_scale=7.5,
                num_images_per_prompt=1
            ).images
            
            generation_time = time.time() - start_time
            
            # 保存
            filename = f"generated_images/test_blog_{test['name']}.png"
            images[0].save(filename)
            
            print(f"      ✅ 成功 ({generation_time:.2f}秒) - {filename}")
            
            results.append({
                'type': 'blog',
                'name': test['name'],
                'success': True,
                'time': generation_time,
                'file': filename,
                'size': images[0].size
            })
            
        except Exception as e:
            print(f"      ❌ 失敗: {e}")
            results.append({
                'type': 'blog',
                'name': test['name'],
                'success': False,
                'error': str(e)
            })
    
    return results

def test_icon_generation(pipe):
    """アイコン生成テスト"""
    print("\n🎨 アイコン生成テスト...")
    
    test_prompts = [
        {
            "name": "tech_minimal",
            "prompt": "Technology innovation symbol with geometric shapes, minimalist design",
            "style": "minimalist"
        },
        {
            "name": "creative_modern",
            "prompt": "Creative arts symbol with modern design elements",
            "style": "modern"
        }
    ]
    
    results = []
    
    for test in test_prompts:
        print(f"\n   📝 テスト: {test['name']}")
        print(f"      プロンプト: {test['prompt'][:50]}...")
        
        try:
            start_time = time.time()
            
            images = pipe(
                prompt=f"{test['prompt']}, logo icon, scalable vector style, professional branding",
                negative_prompt="complex, detailed, photorealistic, cluttered, text, letters",
                height=1024,  # 1:1
                width=1024,
                num_inference_steps=35,
                guidance_scale=9.0,
                num_images_per_prompt=1
            ).images
            
            generation_time = time.time() - start_time
            
            # 保存
            filename = f"generated_images/test_icon_{test['name']}.png"
            images[0].save(filename)
            
            print(f"      ✅ 成功 ({generation_time:.2f}秒) - {filename}")
            
            results.append({
                'type': 'icon',
                'name': test['name'],
                'success': True,
                'time': generation_time,
                'file': filename,
                'size': images[0].size
            })
            
        except Exception as e:
            print(f"      ❌ 失敗: {e}")
            results.append({
                'type': 'icon',
                'name': test['name'],
                'success': False,
                'error': str(e)
            })
    
    return results

def performance_benchmark(pipe):
    """パフォーマンス測定"""
    print("\n⚡ パフォーマンス測定...")
    
    benchmark_configs = [
        {"name": "快速", "steps": 15, "size": (768, 768)},
        {"name": "標準", "steps": 25, "size": (1024, 1024)},
        {"name": "高品質", "steps": 40, "size": (1024, 1024)}
    ]
    
    benchmark_results = []
    test_prompt = "Beautiful mountain landscape, digital art, high quality"
    
    for config in benchmark_configs:
        print(f"\n   ⏱️  {config['name']}設定テスト...")
        print(f"      解像度: {config['size'][0]}×{config['size'][1]}")
        print(f"      ステップ数: {config['steps']}")
        
        times = []
        
        # 3回測定して平均を取る
        for i in range(3):
            try:
                # メモリクリア
                if hasattr(pipe, 'cuda'):
                    torch.cuda.empty_cache()
                gc.collect()
                
                start_time = time.time()
                
                images = pipe(
                    prompt=test_prompt,
                    height=config['size'][1],
                    width=config['size'][0],
                    num_inference_steps=config['steps'],
                    guidance_scale=7.5,
                    num_images_per_prompt=1
                ).images
                
                generation_time = time.time() - start_time
                times.append(generation_time)
                
                print(f"      試行 {i+1}: {generation_time:.2f}秒")
                
            except Exception as e:
                print(f"      試行 {i+1}: 失敗 - {e}")
                break
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"      📊 平均: {avg_time:.2f}秒 (最速: {min_time:.2f}s, 最遅: {max_time:.2f}s)")
            
            benchmark_results.append({
                'config': config['name'],
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'successful_runs': len(times)
            })
        else:
            print(f"      ❌ すべての試行が失敗")
    
    return benchmark_results

def check_memory_usage():
    """メモリ使用量チェック"""
    print("\n📊 メモリ使用量チェック...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            usage_percent = (allocated / total) * 100
            
            print(f"   GPU メモリ:")
            print(f"      使用中: {allocated:.2f}GB")
            print(f"      予約済み: {reserved:.2f}GB")
            print(f"      総容量: {total:.2f}GB")
            print(f"      使用率: {usage_percent:.1f}%")
            
            if usage_percent > 80:
                print("   ⚠️  メモリ使用量が高いです")
            elif usage_percent > 60:
                print("   😐 メモリ使用量は普通です")
            else:
                print("   ✅ メモリ使用量は良好です")
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'usage_percent': usage_percent
            }
        else:
            print("   ❌ CUDA GPU が利用できません")
            return None
            
    except Exception as e:
        print(f"   ❌ メモリチェック失敗: {e}")
        return None

def save_test_results(test_data):
    """テスト結果の保存"""
    print("\n💾 テスト結果保存中...")
    
    try:
        # ログディレクトリ作成
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # タイムスタンプ付きファイル名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"sdxl_test_results_{timestamp}.json"
        
        # 結果をJSONで保存
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"   ✅ テスト結果保存: {log_file}")
        return log_file
        
    except Exception as e:
        print(f"   ❌ 結果保存失敗: {e}")
        return None

def print_summary(all_results):
    """テスト結果サマリー"""
    print("\n" + "=" * 60)
    print("📋 テスト結果サマリー")
    print("=" * 60)
    
    # 全体統計
    total_tests = 0
    successful_tests = 0
    total_time = 0
    
    for category, results in all_results.items():
        if category == 'generation_tests':
            for result in results:
                total_tests += 1
                if result['success']:
                    successful_tests += 1
                    total_time += result.get('time', 0)
    
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    avg_time = total_time / successful_tests if successful_tests > 0 else 0
    
    print(f"🎯 全体結果:")
    print(f"   テスト実行数: {total_tests}")
    print(f"   成功: {successful_tests}")
    print(f"   成功率: {success_rate:.1f}%")
    print(f"   平均生成時間: {avg_time:.2f}秒")
    
    # カテゴリ別結果
    print(f"\n📊 カテゴリ別結果:")
    categories = {}
    for result in all_results.get('generation_tests', []):
        cat = result['type']
        if cat not in categories:
            categories[cat] = {'total': 0, 'success': 0, 'avg_time': 0}
        
        categories[cat]['total'] += 1
        if result['success']:
            categories[cat]['success'] += 1
            categories[cat]['avg_time'] += result.get('time', 0)
    
    for cat, stats in categories.items():
        success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
        avg_time = stats['avg_time'] / stats['success'] if stats['success'] > 0 else 0
        print(f"   {cat.title()}: {stats['success']}/{stats['total']} ({success_rate:.0f}%) - {avg_time:.1f}s平均")
    
    # パフォーマンス結果
    if 'benchmark' in all_results:
        print(f"\n⚡ パフォーマンス:")
        for bench in all_results['benchmark']:
            print(f"   {bench['config']}: {bench['avg_time']:.2f}秒平均")
    
    # メモリ状況
    if 'memory' in all_results and all_results['memory']:
        memory = all_results['memory']
        print(f"\n📊 メモリ使用量: {memory['usage_percent']:.1f}% ({memory['allocated_gb']:.2f}GB/{memory['total_gb']:.1f}GB)")
    
    # 推奨事項
    print(f"\n💡 推奨事項:")
    if success_rate < 80:
        print("   - セットアップを再確認してください")
    if avg_time > 120:
        print("   - より高速な設定を検討してください")
    if all_results.get('memory', {}).get('usage_percent', 0) > 80:
        print("   - メモリ使用量を削減してください")
    
    print("\n📁 生成された画像:")
    print("   generated_images/ フォルダを確認してください")
    
    print("=" * 60)

def main():
    """メイン処理"""
    start_time = time.time()
    
    try:
        print_header()
        
        # セットアップ確認
        if not check_setup():
            return False
        
        # パイプライン初期化
        pipe = initialize_pipeline()
        if pipe is None:
            return False
        
        # テスト実行
        all_results = {
            'test_start_time': datetime.now().isoformat(),
            'generation_tests': [],
            'benchmark': [],
            'memory': None
        }
        
        # 各種生成テスト
        youtube_results = test_youtube_generation(pipe)
        blog_results = test_blog_generation(pipe)
        icon_results = test_icon_generation(pipe)
        
        all_results['generation_tests'].extend(youtube_results)
        all_results['generation_tests'].extend(blog_results)
        all_results['generation_tests'].extend(icon_results)
        
        # パフォーマンステスト
        benchmark_results = performance_benchmark(pipe)
        all_results['benchmark'] = benchmark_results
        
        # メモリチェック
        memory_info = check_memory_usage()
        all_results['memory'] = memory_info
        
        # 結果保存
        total_time = time.time() - start_time
        all_results['total_test_time'] = total_time
        all_results['test_end_time'] = datetime.now().isoformat()
        
        save_test_results(all_results)
        
        # サマリー表示
        print_summary(all_results)
        
        print(f"\n⏱️  総テスト時間: {total_time/60:.1f}分")
        print("🎉 すべてのテストが完了しました！")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️  テストが中断されました")
        return False
    except Exception as e:
        print(f"\n💥 予期しないエラー: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)