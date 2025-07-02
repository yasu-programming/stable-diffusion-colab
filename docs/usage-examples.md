# 使用例とサンプルプロンプト

## YouTubeサムネイル生成例

### 技術系動画サムネイル
```python
prompt = """
modern tech office, programmer working on multiple monitors, 
clean minimalist design, blue and white color scheme, 
professional lighting, high quality, 8k resolution
"""

negative_prompt = """
blurry, low quality, distorted, text, watermark, 
cluttered, messy, dark, unprofessional
"""

# 16:9 アスペクト比
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=896,
    height=512,
    num_inference_steps=20,
    guidance_scale=7.5
).images[0]
```

### 料理・グルメ系動画サムネイル
```python
prompt = """
delicious homemade pasta dish, beautiful food photography, 
warm lighting, cozy kitchen background, appetizing, 
professional food styling, vibrant colors
"""

negative_prompt = """
unappetizing, burnt, messy, low quality, blurry, 
artificial lighting, plastic-looking food
"""
```

### ビジネス・自己啓発系サムネイル
```python
prompt = """
successful business person in modern office, 
confident pose, professional attire, motivational atmosphere, 
clean corporate design, success concept, inspiring
"""

negative_prompt = """
casual clothes, messy background, unprofessional, 
low quality, dark lighting, cluttered
"""
```

## アイコン生成例

### YouTubeチャンネルアイコン
```python
prompt = """
minimalist logo design, simple geometric shapes, 
modern color palette, clean vector style, 
memorable brand symbol, professional
"""

negative_prompt = """
complex details, busy design, low resolution, 
blurry, text, watermark, cluttered
"""

# 1:1 アスペクト比
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=512,
    height=512,
    num_inference_steps=25,
    guidance_scale=8.0
).images[0]
```

### SNSプロフィールアイコン
```python
prompt = """
cute cartoon character, friendly expression, 
simple design, pastel colors, round shape, 
appealing to young audience, kawaii style
"""
```

### ブランドロゴ風アイコン
```python
prompt = """
modern corporate logo, abstract symbol, 
professional design, scalable vector style, 
trustworthy brand image, sophisticated
"""
```

## Note記事アイキャッチ画像例

### テック系記事
```python
prompt = """
abstract technology concept, circuit board patterns, 
futuristic design, blue and purple gradients, 
digital art, clean composition, modern
"""

# 16:9 または 4:3 アスペクト比
width, height = 896, 504  # 16:9
# width, height = 768, 576  # 4:3
```

### ライフスタイル記事
```python
prompt = """
cozy home interior, warm lighting, minimalist style, 
plants and natural elements, peaceful atmosphere, 
lifestyle photography, soft colors
"""
```

### ビジネス記事
```python
prompt = """
modern office space, professional environment, 
growth concept, upward arrows, success symbols, 
clean corporate aesthetic, inspiring
"""
```

## プロンプト作成のコツ

### 効果的なキーワード
- **品質向上**: "high quality", "8k resolution", "professional", "detailed"
- **スタイル指定**: "minimalist", "modern", "clean", "professional"
- **照明**: "soft lighting", "natural light", "studio lighting"
- **色調**: "vibrant colors", "pastel tones", "monochrome"

### ネガティブプロンプトの定番
```python
negative_prompt = """
blurry, low quality, distorted, ugly, deformed, 
text, watermark, signature, bad anatomy, 
cluttered, messy, noise, artifacts
"""
```

### パラメータ調整指針

#### 高品質画像生成
```python
num_inference_steps = 30-50  # 高品質だが時間かかる
guidance_scale = 7-12        # プロンプト忠実度
```

#### 高速生成（Colab最適化）
```python
num_inference_steps = 15-25  # 速度重視
guidance_scale = 6-8         # バランス重視
```

## バッチ生成例

### 複数バリエーション生成
```python
prompts = [
    "modern tech workspace, blue theme",
    "cozy coffee shop, warm lighting", 
    "minimalist office, clean design"
]

for i, prompt in enumerate(prompts):
    image = pipe(prompt=prompt, width=896, height=512).images[0]
    image.save(f"/content/thumbnail_{i+1}.png")
```

## 生成後の後処理

### 画像サイズ調整
```python
from PIL import Image

# サムネイル用リサイズ
image = image.resize((1280, 720), Image.LANCZOS)

# アイコン用リサイズ  
image = image.resize((512, 512), Image.LANCZOS)
```

### 画質最適化
```python
# 高品質保存
image.save("output.png", quality=95, optimize=True)

# ファイルサイズ最適化
image.save("output.jpg", quality=85, optimize=True)
```