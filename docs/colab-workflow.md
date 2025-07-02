# Google Colab ワークフロー

毎回GitHubリポジトリをcloneして作業する際の効率的なワークフローガイド

## 🚀 クイックスタート（3ステップ）

### 1. Colab環境準備
1. [Google Colab](https://colab.research.google.com/) にアクセス
2. **ランタイム** → **ランタイムのタイプを変更** → **GPU (T4)** を選択
3. 新しいノートブックを作成

### 2. リポジトリクローンとセットアップ
```python
# 新しいセルで実行
!git clone https://github.com/YOUR_USERNAME/stable-diffusion-colab.git
%cd stable-diffusion-colab
!python setup.py
```

### 3. 画像生成開始
```python
# クイックテスト
!python quick_start.py
```

## 📋 詳細ワークフロー

### セットアップ（初回のみ5-10分）

#### ステップ1: GPU確認
```python
import torch
print(f"CUDA利用可能: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

#### ステップ2: リポジトリクローン
```python
!git clone https://github.com/YOUR_USERNAME/stable-diffusion-colab.git
%cd stable-diffusion-colab
!ls -la  # ファイル確認
```

#### ステップ3: 自動セットアップ実行
```python
# 全自動セットアップ（5-10分）
!python setup.py

# または手動セットアップ
!pip install -r requirements.txt
!mkdir -p generated_images models_cache temp
```

### 画像生成フェーズ

#### 基本的な画像生成
```python
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

# パイプライン初期化（初回のみ時間がかかる）
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    cache_dir="./models_cache"
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")
    pipe.enable_memory_efficient_attention()
```

#### YouTubeサムネイル生成
```python
prompt = "modern tech office, professional lighting, clean design"
negative_prompt = "blurry, low quality, messy"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=896, height=512,  # 16:9 比率
    num_inference_steps=20,
    guidance_scale=7.5
).images[0]

# 表示と保存
plt.figure(figsize=(12, 6))
plt.imshow(image)
plt.axis("off")
plt.show()

image.save("generated_images/thumbnail.png")
```

#### アイコン生成
```python
prompt = "minimalist logo, simple design, professional"
negative_prompt = "complex, cluttered, text"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=512, height=512,  # 1:1 比率
    num_inference_steps=25,
    guidance_scale=8.0
).images[0]

image.save("generated_images/icon.png")
```

### ファイル管理

#### 生成画像の一括ダウンロード
```python
import zipfile
import os
from google.colab import files

# ZIPファイル作成
with zipfile.ZipFile('generated_images.zip', 'w') as zipf:
    for root, dirs, filenames in os.walk('generated_images'):
        for filename in filenames:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                zipf.write(os.path.join(root, filename), filename)

# ダウンロード
files.download('generated_images.zip')
```

#### Google Driveマウント（オプション）
```python
from google.colab import drive
drive.mount('/content/drive')

# Drive保存
image.save('/content/drive/MyDrive/ai_images/thumbnail.png')
```

## ⚡ 効率化のコツ

### 1. テンプレートノートブック使用
- `colab_quickstart.ipynb` をコピーして使用
- 必要な部分のみ実行

### 2. バッチ生成
```python
prompts = [
    "tech office, blue theme",
    "cozy cafe, warm lighting", 
    "modern workspace, minimal"
]

for i, prompt in enumerate(prompts):
    image = pipe(prompt=prompt, width=896, height=512).images[0]
    image.save(f"generated_images/batch_{i+1}.png")
```

### 3. メモリ管理
```python
# 不要な変数削除
del pipe
torch.cuda.empty_cache()

# 必要時に再初期化
pipe = StableDiffusionPipeline.from_pretrained(...)
```

### 4. セッション時間管理
- 無料版は約12時間制限
- 定期的にファイルをダウンロード
- 長時間作業時はGoogle Driveに自動保存

## 🔧 トラブルシューティング

### GPU利用不可の場合
```python
# CPU使用（非常に遅い）
pipe = pipe.to("cpu")
# または
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float32  # CPU用
)
```

### メモリ不足エラー
```python
# 解像度を下げる
width, height = 512, 512  # より小さく

# バッチサイズを1に
# 複数画像生成時は1つずつ実行
```

### モデルダウンロードエラー
```python
# キャッシュクリア
!rm -rf ./models_cache/*

# 手動ダウンロード
!huggingface-cli download runwayml/stable-diffusion-v1-5
```

## 📁 ファイル構成

セットアップ後のディレクトリ構造：
```
stable-diffusion-colab/
├── setup.py              # 自動セットアップスクリプト
├── quick_start.py         # クイックテスト用
├── requirements.txt       # 依存関係
├── colab_quickstart.ipynb # テンプレートノートブック
├── docs/                  # ドキュメント
├── generated_images/      # 生成画像保存先
├── models_cache/          # モデルキャッシュ
└── temp/                  # 一時ファイル
```

このワークフローにより、毎回のセットアップ時間を最小化し、すぐに画像生成作業に入れます。