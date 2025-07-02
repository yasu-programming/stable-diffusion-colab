# Google Colab セットアップガイド

## 事前準備

### 1. Google Colabアクセス
- [Google Colab](https://colab.research.google.com/) にアクセス
- Googleアカウントでログイン

### 2. GPU設定
```python
# ランタイムタイプを確認
!nvidia-smi
```
- メニュー → 「ランタイム」→「ランタイムのタイプを変更」
- ハードウェア アクセラレータ: **GPU (T4)**を選択

## インストール手順

### Step 1: 必要なライブラリのインストール
```python
# 基本ライブラリ
!pip install diffusers transformers accelerate safetensors
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 画像処理
!pip install pillow numpy matplotlib

# 日本語対応（オプション）
!pip install fugashi unidic-lite
```

### Step 2: モデルのダウンロード
```python
from diffusers import StableDiffusionPipeline
import torch

# 商用利用可能なモデル
model_id = "runwayml/stable-diffusion-v1-5"  # MIT License
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
)
pipe = pipe.to("cuda")
```

### Step 3: メモリ最適化
```python
# メモリ効率化
pipe.enable_memory_efficient_attention()
pipe.enable_model_cpu_offload()

# VAE Slicing (大きな画像用)
pipe.enable_vae_slicing()
```

## 使用上の注意

### GPU制限
- 無料版は約12時間の制限
- 連続使用でセッション切断の可能性
- 定期的にファイルをダウンロード保存

### メモリ管理
- 大量画像生成時はバッチ処理を避ける
- 不要な変数は`del`で削除
- `torch.cuda.empty_cache()`でメモリクリア

### ファイル保存
```python
# Google Driveマウント
from google.colab import drive
drive.mount('/content/drive')

# 画像保存先
save_path = "/content/drive/MyDrive/generated_images/"
```