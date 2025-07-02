# SDXL 使用例とサンプルコード

Stable Diffusion XL Base 1.0を使用したYouTube・Note用画像生成の具体的なサンプルコードと実践例です。

## 🎬 YouTube サムネイル生成例

### ゲーミング系チャンネル向け

```python
import torch
from diffusers import DiffusionPipeline
from PIL import Image
import os

# SDXL パイプラインの初期化
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

def generate_gaming_thumbnail(game_title, mood="exciting"):
    """ゲーミング系YouTubeサムネイル生成"""
    
    mood_styles = {
        "exciting": "vibrant colors, dynamic action, high energy, RGB lighting",
        "mysterious": "dark atmosphere, neon lights, cyberpunk style, dramatic shadows",
        "competitive": "intense focus, professional gaming setup, tournament atmosphere"
    }
    
    base_prompt = f"""
    {game_title} gaming setup with professional gamer, {mood_styles[mood]}, 
    modern gaming chair, multiple monitors, mechanical keyboard, 
    gaming headphones, excited facial expression, YouTube thumbnail style,
    professional photography, ultra detailed, 8K resolution
    """
    
    negative_prompt = """
    blurry, low quality, boring, plain, text overlay, watermark,
    sad, depressing, old equipment, cluttered, messy
    """
    
    images = pipe(
        prompt=base_prompt,
        negative_prompt=negative_prompt,
        height=576,  # 16:9 aspect ratio
        width=1024,
        num_inference_steps=30,
        guidance_scale=8.0,
        num_images_per_prompt=3
    ).images
    
    return images

# 使用例
minecraft_thumbnails = generate_gaming_thumbnail("Minecraft", "exciting")
apex_thumbnails = generate_gaming_thumbnail("Apex Legends", "competitive")

# 保存
os.makedirs("youtube_thumbnails", exist_ok=True)
for i, img in enumerate(minecraft_thumbnails):
    img.save(f"youtube_thumbnails/minecraft_thumbnail_{i+1}.png")
```

### 教育系チャンネル向け

```python
def generate_educational_thumbnail(subject, style="professional"):
    """教育系YouTubeサムネイル生成"""
    
    style_prompts = {
        "professional": "clean, modern, professional presentation, bright lighting",
        "friendly": "warm, approachable, colorful, friendly atmosphere",
        "academic": "scholarly, serious, library setting, academic atmosphere"
    }
    
    subject_contexts = {
        "programming": "laptop with code, programming books, clean desk setup",
        "language": "books, world map, language learning materials",
        "science": "laboratory equipment, charts, diagrams, scientific tools",
        "business": "office setting, charts, professional documents"
    }
    
    prompt = f"""
    {subject} educational content, teacher or instructor, 
    {subject_contexts.get(subject, "educational materials")},
    {style_prompts[style]}, clear composition, engaging presentation,
    YouTube educational thumbnail, professional quality, detailed
    """
    
    negative_prompt = """
    cluttered, confusing, dark, unprofessional, low quality,
    distracting elements, text overlay, watermark
    """
    
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=576, width=1024,
        num_inference_steps=25,
        guidance_scale=7.5,
        num_images_per_prompt=2
    ).images

# 各教科の例
programming_thumb = generate_educational_thumbnail("programming", "professional")
language_thumb = generate_educational_thumbnail("language", "friendly")
```

### Vlog系チャンネル向け

```python
def generate_vlog_thumbnail(theme, setting="indoor"):
    """Vlog系YouTubeサムネイル生成"""
    
    settings = {
        "indoor": "cozy room, warm lighting, comfortable furniture",
        "outdoor": "natural landscape, golden hour lighting, scenic background",
        "urban": "city background, modern architecture, street style"
    }
    
    themes = {
        "daily": "casual, relaxed, everyday life, natural expression",
        "travel": "adventure, exploration, excited expression, travel gear",
        "lifestyle": "trendy, fashionable, instagram-worthy, aesthetic"
    }
    
    prompt = f"""
    Young content creator, {themes[theme]}, {settings[setting]},
    engaging with camera, authentic smile, vlog style content,
    natural lighting, high quality portrait, YouTube vlog thumbnail,
    professional photography, detailed, appealing composition
    """
    
    negative_prompt = """
    artificial, posed, low quality, blurry, dark, unappealing,
    professional studio lighting, overly edited, fake
    """
    
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=576, width=1024,
        num_inference_steps=28,
        guidance_scale=7.0,
        num_images_per_prompt=4
    ).images

# 使用例
daily_vlog = generate_vlog_thumbnail("daily", "indoor")
travel_vlog = generate_vlog_thumbnail("travel", "outdoor")
```

## 📝 Note・ブログ記事用画像生成例

### テック系記事向け

```python
def generate_tech_blog_image(topic, style="modern"):
    """テック系ブログ記事用画像生成"""
    
    tech_topics = {
        "ai": "artificial intelligence, neural networks, futuristic technology",
        "web_dev": "web development, coding, modern laptop, clean workspace",
        "mobile": "smartphone, mobile app interface, modern device",
        "blockchain": "cryptocurrency, blockchain concept, digital finance",
        "cloud": "cloud computing, server infrastructure, digital networks"
    }
    
    styles = {
        "modern": "clean, minimalist, professional, contemporary design",
        "futuristic": "sci-fi, neon colors, holographic elements, cyber aesthetic",
        "corporate": "business professional, corporate environment, formal"
    }
    
    prompt = f"""
    {tech_topics[topic]}, {styles[style]}, blog header image,
    professional tech photography, soft lighting, high quality,
    1.91:1 aspect ratio, clean composition, no text
    """
    
    negative_prompt = """
    cluttered, messy, outdated technology, low quality,
    unprofessional, amateur, text overlay, watermark
    """
    
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=536,  # 1.91:1 aspect ratio for blog headers
        width=1024,
        num_inference_steps=30,
        guidance_scale=7.5,
        num_images_per_prompt=2
    ).images

# 各トピックの例
ai_article = generate_tech_blog_image("ai", "futuristic")
webdev_article = generate_tech_blog_image("web_dev", "modern")
```

### ライフスタイル記事向け

```python
def generate_lifestyle_image(category, mood="bright"):
    """ライフスタイル記事用画像生成"""
    
    categories = {
        "wellness": "yoga, meditation, healthy lifestyle, peaceful atmosphere",
        "productivity": "organized workspace, planning, efficient work setup",
        "cooking": "modern kitchen, fresh ingredients, cooking preparation",
        "home": "cozy home interior, comfortable living space, home decor",
        "fitness": "exercise equipment, workout space, healthy lifestyle"
    }
    
    moods = {
        "bright": "bright, cheerful, optimistic, clean lighting",
        "cozy": "warm, comfortable, inviting, soft lighting",
        "elegant": "sophisticated, refined, luxury, premium quality"
    }
    
    prompt = f"""
    {categories[category]}, {moods[mood]}, lifestyle photography,
    Instagram-worthy, aesthetic composition, natural lighting,
    blog article header, professional quality, detailed
    """
    
    negative_prompt = """
    dark, gloomy, cluttered, unprofessional, low quality,
    chaotic, messy, artificial lighting
    """
    
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=536, width=1024,
        num_inference_steps=25,
        guidance_scale=7.0,
        num_images_per_prompt=3
    ).images

# 使用例
wellness_image = generate_lifestyle_image("wellness", "bright")
productivity_image = generate_lifestyle_image("productivity", "cozy")
```

## 🎨 アイコン・ロゴ生成例

### ブランドアイコン

```python
def generate_brand_icon(brand_type, style="minimalist"):
    """ブランドアイコン生成"""
    
    brand_types = {
        "tech": "technology, innovation, digital, modern geometric shapes",
        "creative": "art, creativity, design, abstract artistic elements",
        "finance": "trust, stability, growth, professional geometric design",
        "health": "wellness, care, medical, clean simple symbols",
        "food": "culinary, fresh, organic, food-related symbols"
    }
    
    styles = {
        "minimalist": "simple, clean lines, minimal design, single color",
        "modern": "contemporary, sleek, professional, subtle gradients",
        "playful": "fun, colorful, friendly, approachable design"
    }
    
    prompt = f"""
    {brand_types[brand_type]} logo icon, {styles[style]},
    scalable vector style, professional branding, clean background,
    symbol design, corporate identity, high quality, simple composition
    """
    
    negative_prompt = """
    complex, detailed, photorealistic, cluttered, busy design,
    multiple colors, text, letters, shadows, 3D effects
    """
    
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=1024, width=1024,  # Square format for icons
        num_inference_steps=35,
        guidance_scale=9.0,
        num_images_per_prompt=6
    ).images

# 使用例
tech_icons = generate_brand_icon("tech", "minimalist")
creative_icons = generate_brand_icon("creative", "modern")
```

### SNSアイコン

```python
def generate_social_icon(personality, platform="general"):
    """SNS用プロフィールアイコン生成"""
    
    personalities = {
        "professional": "business professional, confident, approachable, corporate style",
        "creative": "artistic, expressive, colorful, creative professional",
        "friendly": "warm, welcoming, casual, personable",
        "expert": "knowledgeable, trustworthy, authority figure, professional"
    }
    
    platform_specs = {
        "twitter": "profile photo style, clear facial features, good contrast",
        "linkedin": "professional headshot, business appropriate, formal",
        "instagram": "lifestyle, aesthetic, visually appealing",
        "general": "versatile, suitable for multiple platforms"
    }
    
    prompt = f"""
    {personalities[personality]}, {platform_specs[platform]},
    avatar style, clean background, professional quality,
    social media profile picture, high resolution, detailed
    """
    
    negative_prompt = """
    blurry, low quality, inappropriate, unprofessional,
    distracting background, poor lighting, amateur
    """
    
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=1024, width=1024,
        num_inference_steps=30,
        guidance_scale=8.0,
        num_images_per_prompt=4
    ).images

# 使用例
linkedin_avatar = generate_social_icon("professional", "linkedin")
twitter_avatar = generate_social_icon("friendly", "twitter")
```

## 🔄 バッチ処理の実践例

### 大量画像生成システム

```python
import json
import time
from datetime import datetime
import os

class SDXLBatchGenerator:
    def __init__(self, pipe):
        self.pipe = pipe
        self.generation_log = []
    
    def generate_youtube_series(self, series_name, episode_count=10):
        """YouTubeシリーズ用サムネイル一括生成"""
        
        # シリーズ用プロンプトテンプレート
        series_templates = {
            "programming_tutorial": [
                "Python programming tutorial, coding on laptop, episode {ep}",
                "Web development lesson, HTML CSS JavaScript, tutorial {ep}",
                "Database design tutorial, SQL learning, lesson {ep}"
            ],
            "gaming_playthrough": [
                "Gaming session {ep}, exciting gameplay, action scene",
                "Boss battle episode {ep}, intense gaming moment",
                "New level exploration, gaming adventure episode {ep}"
            ],
            "lifestyle_vlog": [
                "Daily vlog episode {ep}, casual lifestyle content",
                "Morning routine vlog {ep}, healthy lifestyle",
                "Weekend activities vlog {ep}, fun and relaxation"
            ]
        }
        
        templates = series_templates.get(series_name, series_templates["programming_tutorial"])
        output_dir = f"youtube_series_{series_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)
        
        for ep in range(1, episode_count + 1):
            template = templates[(ep - 1) % len(templates)]
            prompt = template.format(ep=ep)
            
            full_prompt = f"""
            {prompt}, YouTube thumbnail style, professional quality,
            engaging composition, vibrant colors, high energy,
            ultra detailed, 8K resolution
            """
            
            print(f"Generating episode {ep}: {prompt[:50]}...")
            
            try:
                start_time = time.time()
                
                images = self.pipe(
                    prompt=full_prompt,
                    negative_prompt="blurry, low quality, boring, text overlay",
                    height=576, width=1024,
                    num_inference_steps=25,
                    guidance_scale=8.0,
                    num_images_per_prompt=2
                ).images
                
                generation_time = time.time() - start_time
                
                # 保存
                for i, img in enumerate(images):
                    filename = f"episode_{ep:02d}_variant_{i+1}.png"
                    img.save(os.path.join(output_dir, filename))
                
                # ログ記録
                self.generation_log.append({
                    'episode': ep,
                    'prompt': prompt,
                    'generation_time': generation_time,
                    'success': True,
                    'image_count': len(images)
                })
                
                print(f"✅ Episode {ep} completed in {generation_time:.2f}s")
                
                # メモリクリア
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Episode {ep} failed: {e}")
                self.generation_log.append({
                    'episode': ep,
                    'prompt': prompt,
                    'error': str(e),
                    'success': False
                })
        
        # 結果サマリー保存
        summary = {
            'series_name': series_name,
            'total_episodes': episode_count,
            'successful_generations': sum(1 for log in self.generation_log if log['success']),
            'total_time': sum(log.get('generation_time', 0) for log in self.generation_log),
            'output_directory': output_dir,
            'generation_log': self.generation_log
        }
        
        with open(os.path.join(output_dir, 'generation_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return summary

# 使用例
batch_generator = SDXLBatchGenerator(pipe)
result = batch_generator.generate_youtube_series("programming_tutorial", 5)
print(f"Generated {result['successful_generations']} thumbnails in {result['total_time']:.2f}s")
```

### カスタムプロンプト一括処理

```python
def batch_custom_prompts(prompts_file, pipe, output_dir="custom_batch"):
    """カスタムプロンプトファイルから一括生成"""
    
    # プロンプトファイル読み込み
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for i, prompt_config in enumerate(prompts_data):
        print(f"Processing {i+1}/{len(prompts_data)}: {prompt_config['name']}")
        
        try:
            images = pipe(**prompt_config['config']).images
            
            # 画像保存
            for j, img in enumerate(images):
                filename = f"{prompt_config['name']}_{j+1}.png"
                img.save(os.path.join(output_dir, filename))
            
            results.append({
                'name': prompt_config['name'],
                'success': True,
                'image_count': len(images)
            })
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                'name': prompt_config['name'],
                'success': False,
                'error': str(e)
            })
    
    return results

# プロンプトファイル例（JSON形式）
sample_prompts = [
    {
        "name": "tech_blog_ai",
        "config": {
            "prompt": "Artificial intelligence concept, futuristic technology, blog header",
            "height": 536, "width": 1024,
            "num_inference_steps": 30,
            "guidance_scale": 7.5
        }
    },
    {
        "name": "youtube_gaming", 
        "config": {
            "prompt": "Gaming setup with RGB lighting, YouTube thumbnail",
            "height": 576, "width": 1024,
            "num_inference_steps": 25,
            "guidance_scale": 8.0
        }
    }
]

# プロンプトファイル保存
with open('custom_prompts.json', 'w', encoding='utf-8') as f:
    json.dump(sample_prompts, f, ensure_ascii=False, indent=2)

# バッチ実行
batch_results = batch_custom_prompts('custom_prompts.json', pipe)
```

## 📊 品質管理とA/Bテスト

### サムネイル効果測定

```python
def generate_ab_test_thumbnails(base_prompt, variations):
    """A/Bテスト用サムネイル生成"""
    
    results = {}
    
    for variant_name, modifications in variations.items():
        modified_prompt = f"{base_prompt}, {modifications}"
        
        print(f"Generating variant: {variant_name}")
        
        images = pipe(
            prompt=modified_prompt,
            height=576, width=1024,
            num_inference_steps=30,
            guidance_scale=8.0,
            num_images_per_prompt=3
        ).images
        
        results[variant_name] = {
            'prompt': modified_prompt,
            'images': images
        }
        
        # 保存
        variant_dir = f"ab_test_{variant_name}"
        os.makedirs(variant_dir, exist_ok=True)
        
        for i, img in enumerate(images):
            img.save(f"{variant_dir}/variant_{i+1}.png")
    
    return results

# A/Bテスト例
base_prompt = "Gaming tutorial YouTube thumbnail, excited gamer"

variants = {
    "bright_colorful": "vibrant colors, high energy, RGB lighting",
    "dark_mysterious": "dark atmosphere, neon accents, mysterious mood",
    "clean_professional": "clean setup, professional lighting, minimalist"
}

ab_results = generate_ab_test_thumbnails(base_prompt, variants)
```

このサンプルコードを参考に、目的に応じたSDXL画像生成を効率的に行ってください。