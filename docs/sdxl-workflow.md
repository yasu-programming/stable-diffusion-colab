# SDXL ワークフローガイド

Stable Diffusion XL Base 1.0を使用したYouTube・Note用画像生成の実践的なワークフローとベストプラクティスです。

## 🎯 SDXL ワークフローの概要

### 基本ワークフロー
1. **プロンプト設計** → 目的に応じた詳細なプロンプト作成
2. **パラメータ調整** → 用途別の最適設定
3. **初期生成** → ベース画像の生成
4. **品質評価** → 結果の確認と改善点特定
5. **リファイン** → 必要に応じた再生成・調整
6. **後処理** → フォーマット変換・サイズ調整

### SDXL特有の特徴
- **高解像度標準**: 1024×1024が基本
- **詳細なプロンプト理解**: 複雑な指示にも対応
- **安定した品質**: 一貫性のある高品質出力
- **柔軟なアスペクト比**: 様々な比率に対応

## 📝 プロンプト設計戦略

### SDXLに最適化されたプロンプト構造

```python
# プロンプトテンプレート
template = """
{main_subject}, {style_description}, {technical_specs}, {quality_modifiers}
"""

# 例：YouTubeサムネイル用
youtube_prompt = """
A vibrant gaming setup with RGB lighting, modern gaming chair and multiple monitors, 
cyberpunk style, neon colors, high contrast, professional photography, 
ultra detailed, 8K resolution, trending on artstation
"""

# 例：Note記事用
note_prompt = """
A minimalist workspace with laptop, coffee cup and notebook on wooden desk, 
natural lighting from window, soft shadows, clean composition, 
professional stock photo style, high quality, commercial photography
"""
```

### 効果的なプロンプト要素

#### 1. 主要被写体の明確化
```python
subjects = [
    "A professional businessman",
    "Modern smartphone on desk", 
    "Cozy living room interior",
    "Abstract geometric pattern"
]
```

#### 2. スタイル指定
```python
styles = [
    "professional photography",
    "digital art illustration", 
    "minimalist design",
    "vintage aesthetic",
    "cyberpunk style"
]
```

#### 3. 品質向上キーワード
```python
quality_terms = [
    "ultra detailed", "8K resolution", "high quality",
    "professional lighting", "sharp focus", "vivid colors",
    "trending on artstation", "award winning"
]
```

## ⚙️ 用途別パラメータ設定

### YouTube サムネイル生成

```python
def generate_youtube_thumbnail(prompt, pipe):
    """YouTube サムネイル用画像生成"""
    config = {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, distorted, watermark, text",
        "height": 576,           # 16:9 比率（1024x576）
        "width": 1024,
        "num_inference_steps": 25,
        "guidance_scale": 8.0,   # 強めのガイダンス
        "num_images_per_prompt": 4,  # 複数バリエーション
    }
    
    images = pipe(**config).images
    return images

# 使用例
youtube_prompt = """
Excited gamer with headphones playing video game, colorful RGB background, 
dynamic action scene, vibrant colors, professional gaming photography, 
high energy, dramatic lighting, ultra detailed
"""

thumbnail_images = generate_youtube_thumbnail(youtube_prompt, pipe)
```

### Note・ブログ記事用画像

```python
def generate_blog_image(prompt, pipe):
    """ブログ記事用アイキャッチ画像生成"""
    config = {
        "prompt": prompt,
        "negative_prompt": "cluttered, messy, low quality, amateur",
        "height": 536,           # 1.91:1 比率（1024x536）
        "width": 1024,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "num_images_per_prompt": 2,
    }
    
    images = pipe(**config).images
    return images

# 使用例
blog_prompt = """
Clean modern office workspace with laptop, notebook and coffee, 
natural lighting, minimalist design, productivity concept, 
professional stock photography, soft colors, high quality
"""

blog_images = generate_blog_image(blog_prompt, pipe)
```

### アイコン・ロゴ用画像

```python
def generate_icon(prompt, pipe):
    """アイコン・ロゴ用画像生成"""
    config = {
        "prompt": prompt,
        "negative_prompt": "complex, detailed background, realistic, photographic",
        "height": 1024,          # 正方形
        "width": 1024,
        "num_inference_steps": 35,
        "guidance_scale": 9.0,   # 最強ガイダンス
        "num_images_per_prompt": 6,  # 多数生成で選択
    }
    
    images = pipe(**config).images
    return images

# 使用例
icon_prompt = """
Simple geometric logo design, abstract symbol, clean lines, 
modern minimalist style, single color on white background, 
vector art style, professional branding, scalable design
"""

icon_images = generate_icon(icon_prompt, pipe)
```

## 🔄 バッチ処理ワークフロー

### 効率的な大量生成

```python
def batch_generate_images(prompts, pipe, output_dir="generated_images"):
    """複数プロンプトの効率的なバッチ処理"""
    import os
    from datetime import datetime
    
    # 出力ディレクトリ作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(output_dir, f"batch_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)
    
    results = []
    
    for i, prompt_config in enumerate(prompts):
        print(f"Processing {i+1}/{len(prompts)}: {prompt_config['name']}")
        
        try:
            # メモリクリア
            torch.cuda.empty_cache()
            
            # 画像生成
            images = pipe(**prompt_config['config']).images
            
            # 保存
            for j, image in enumerate(images):
                filename = f"{prompt_config['name']}_{j+1}.png"
                filepath = os.path.join(batch_dir, filename)
                image.save(filepath)
                results.append({
                    'prompt': prompt_config['name'],
                    'file': filepath,
                    'config': prompt_config['config']
                })
            
            print(f"✅ Saved {len(images)} images")
            
        except Exception as e:
            print(f"❌ Error processing {prompt_config['name']}: {e}")
            continue
    
    return results, batch_dir

# 使用例
batch_prompts = [
    {
        "name": "tech_thumbnail_1",
        "config": {
            "prompt": "Modern laptop with coding screen, programmer workspace, tech setup",
            "height": 576, "width": 1024,
            "num_inference_steps": 25,
            "num_images_per_prompt": 2
        }
    },
    {
        "name": "business_blog_1", 
        "config": {
            "prompt": "Professional meeting room, business discussion, modern office",
            "height": 536, "width": 1024,
            "num_inference_steps": 30,
            "num_images_per_prompt": 2
        }
    }
]

results, output_dir = batch_generate_images(batch_prompts, pipe)
```

## 🎨 クリエイティブ技法

### プロンプトの段階的改良

```python
def iterative_prompt_refinement(base_prompt, pipe, iterations=3):
    """プロンプトを段階的に改良して最適化"""
    
    # 基本プロンプトから開始
    current_prompt = base_prompt
    results = []
    
    refinements = [
        # 1回目：基本的な品質向上
        ", high quality, detailed, professional",
        # 2回目：スタイル追加
        ", digital art, trending on artstation, award winning",
        # 3回目：技術的仕様追加
        ", ultra detailed, 8K resolution, perfect composition, dramatic lighting"
    ]
    
    for i, refinement in enumerate(refinements[:iterations]):
        refined_prompt = current_prompt + refinement
        
        print(f"Iteration {i+1}: {refined_prompt[:100]}...")
        
        images = pipe(
            prompt=refined_prompt,
            num_inference_steps=25,
            guidance_scale=7.5,
            num_images_per_prompt=2
        ).images
        
        results.append({
            'iteration': i+1,
            'prompt': refined_prompt,
            'images': images
        })
        
        current_prompt = refined_prompt
    
    return results

# 使用例
base = "A cozy coffee shop interior with warm lighting"
refinement_results = iterative_prompt_refinement(base, pipe)
```

### ネガティブプロンプトの活用

```python
# 用途別ネガティブプロンプト集
negative_prompts = {
    "youtube_thumbnail": """
        blurry, low quality, boring, plain, amateur, watermark, text overlay,
        dark, gloomy, sad, depressing, low energy, static, old fashioned
    """,
    
    "blog_image": """
        cluttered, messy, chaotic, unprofessional, low quality, distorted,
        harsh lighting, oversaturated, cartoonish, childish
    """,
    
    "icon_logo": """
        complex background, photorealistic, detailed texture, shadows,
        multiple colors, busy design, text, letters, numbers
    """,
    
    "general": """
        blurry, low quality, distorted, deformed, ugly, bad anatomy,
        watermark, signature, low resolution, pixelated
    """
}

def generate_with_negative(prompt, category, pipe):
    """ネガティブプロンプト付きで生成"""
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompts.get(category, negative_prompts["general"]),
        num_inference_steps=30,
        guidance_scale=8.0
    ).images[0]
```

## 📊 品質管理とワークフロー最適化

### 生成品質の評価

```python
def evaluate_image_quality(images, criteria):
    """生成画像の品質評価（主観的基準）"""
    
    evaluation = {
        'composition': [],    # 構図の良さ
        'clarity': [],       # 鮮明さ
        'relevance': [],     # プロンプトとの関連性
        'appeal': []         # 視覚的魅力
    }
    
    for i, image in enumerate(images):
        print(f"Image {i+1} evaluation:")
        print("Rate each aspect (1-5):")
        
        for criterion in criteria:
            while True:
                try:
                    score = int(input(f"{criterion}: "))
                    if 1 <= score <= 5:
                        evaluation[criterion].append(score)
                        break
                except ValueError:
                    print("Please enter a number between 1-5")
    
    # 平均スコア計算
    averages = {k: sum(v)/len(v) for k, v in evaluation.items()}
    return evaluation, averages

# 自動品質チェック（技術的指標）
def technical_quality_check(image):
    """技術的品質指標の確認"""
    import numpy as np
    from PIL import ImageStat
    
    # 基本統計
    stat = ImageStat.Stat(image)
    
    metrics = {
        'mean_brightness': np.mean(stat.mean),
        'contrast': np.std(stat.mean),
        'size': image.size,
        'mode': image.mode
    }
    
    return metrics
```

### 最適パラメータの発見

```python
def parameter_optimization(prompt, pipe, param_ranges):
    """パラメータの組み合わせをテストして最適化"""
    
    import itertools
    
    # パラメータの組み合わせ生成
    param_combinations = list(itertools.product(*param_ranges.values()))
    param_names = list(param_ranges.keys())
    
    results = []
    
    for i, combination in enumerate(param_combinations):
        config = dict(zip(param_names, combination))
        config['prompt'] = prompt
        
        print(f"Testing combination {i+1}/{len(param_combinations)}: {config}")
        
        try:
            image = pipe(**config).images[0]
            
            # 技術的品質評価
            quality = technical_quality_check(image)
            
            results.append({
                'config': config,
                'image': image,
                'quality': quality,
                'score': quality['contrast']  # 仮の評価指標
            })
            
        except Exception as e:
            print(f"Error with config {config}: {e}")
    
    # 最高スコアの設定を返す
    best_result = max(results, key=lambda x: x['score'])
    return best_result, results

# 使用例：YouTubeサムネイル最適化
optimization_ranges = {
    'num_inference_steps': [20, 25, 30],
    'guidance_scale': [7.0, 7.5, 8.0, 8.5],
    'height': [576],
    'width': [1024]
}

best_config, all_results = parameter_optimization(
    "Exciting gaming setup with RGB lighting",
    pipe,
    optimization_ranges
)
```

## 🔧 トラブルシューティングワークフロー

### よくある問題の対処フロー

```python
def troubleshoot_generation(pipe, prompt, config):
    """生成エラーの段階的トラブルシューティング"""
    
    troubleshooting_steps = [
        {
            'name': 'Memory cleanup',
            'action': lambda: (torch.cuda.empty_cache(), gc.collect()),
            'description': 'GPU・RAMメモリのクリア'
        },
        {
            'name': 'Reduce resolution',
            'action': lambda: config.update({'height': 768, 'width': 768}),
            'description': '解像度を下げてメモリ使用量削減'
        },
        {
            'name': 'Reduce batch size',
            'action': lambda: config.update({'num_images_per_prompt': 1}),
            'description': 'バッチサイズを1に削減'
        },
        {
            'name': 'Reduce inference steps',
            'action': lambda: config.update({'num_inference_steps': 15}),
            'description': '推論ステップ数を削減'
        },
        {
            'name': 'Enable sequential offload',
            'action': lambda: pipe.enable_sequential_cpu_offload(),
            'description': 'より積極的なCPUオフロード'
        }
    ]
    
    for step in troubleshooting_steps:
        print(f"Trying: {step['name']} - {step['description']}")
        
        try:
            step['action']()
            
            # 生成テスト
            test_image = pipe(
                prompt=prompt,
                **{k: v for k, v in config.items() if k != 'prompt'}
            ).images[0]
            
            print(f"✅ Success with {step['name']}")
            return test_image, config
            
        except Exception as e:
            print(f"❌ {step['name']} failed: {e}")
            continue
    
    print("💥 All troubleshooting steps failed")
    return None, config
```

## 📈 ワークフロー分析と改善

### 生成効率の追跡

```python
import time
import json
from datetime import datetime

class WorkflowTracker:
    def __init__(self):
        self.sessions = []
        self.current_session = None
    
    def start_session(self, session_name):
        """新しい作業セッションを開始"""
        self.current_session = {
            'name': session_name,
            'start_time': datetime.now(),
            'generations': [],
            'total_time': 0,
            'success_rate': 0
        }
    
    def log_generation(self, prompt, config, success, generation_time):
        """個別の生成を記録"""
        if self.current_session:
            self.current_session['generations'].append({
                'prompt': prompt[:50] + '...',
                'config': config,
                'success': success,
                'time': generation_time,
                'timestamp': datetime.now().isoformat()
            })
    
    def end_session(self):
        """セッションを終了して統計を計算"""
        if self.current_session:
            generations = self.current_session['generations']
            
            self.current_session['total_time'] = sum(g['time'] for g in generations)
            self.current_session['success_rate'] = sum(g['success'] for g in generations) / len(generations) if generations else 0
            self.current_session['end_time'] = datetime.now()
            
            self.sessions.append(self.current_session)
            self.current_session = None
    
    def get_statistics(self):
        """全体的な統計を取得"""
        if not self.sessions:
            return "No sessions recorded"
        
        total_generations = sum(len(s['generations']) for s in self.sessions)
        avg_success_rate = sum(s['success_rate'] for s in self.sessions) / len(self.sessions)
        avg_time_per_image = sum(s['total_time'] for s in self.sessions) / total_generations if total_generations > 0 else 0
        
        return {
            'total_sessions': len(self.sessions),
            'total_generations': total_generations,
            'average_success_rate': avg_success_rate,
            'average_time_per_image': avg_time_per_image
        }

# 使用例
tracker = WorkflowTracker()
tracker.start_session("YouTube thumbnails batch")

# 生成処理の例
start_time = time.time()
try:
    image = pipe(prompt="Gaming setup", height=576, width=1024).images[0]
    success = True
except:
    success = False
generation_time = time.time() - start_time

tracker.log_generation("Gaming setup", {"height": 576, "width": 1024}, success, generation_time)
tracker.end_session()

print(tracker.get_statistics())
```

このワークフローガイドを参考に、効率的なSDXL画像生成を実践してください。次は[SDXL 使用例](sdxl-examples.md)で具体的なサンプルコードを確認できます。