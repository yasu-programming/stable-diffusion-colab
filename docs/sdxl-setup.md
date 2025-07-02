# SDXL Base 1.0 Google Colab セットアップガイド

Stable Diffusion XL Base 1.0をGoogle Colab環境で商用利用可能な画像生成システムとして構築する詳細手順です。

## 🎯 SDXL Base 1.0 について

### モデル仕様
- **正式名称**: Stable Diffusion XL Base 1.0
- **開発者**: Stability AI
- **ライセンス**: CreativeML Open RAIL++-M
- **標準解像度**: 1024×1024
- **商用利用**: 可能（アート・教育・研究用途）

### Hugging Face公式推奨事項
- **diffusers**: >= 0.19.0
- **PyTorch**: 2.0以上
- **CUDA**: 対応GPU必須
- **精度**: float16推奨
- **メモリ効率化**: CPU オフロード推奨

## 🚀 Google Colab 環境構築

### ステップ1: Colab環境の準備

```python
# GPU使用確認とCUDAバージョン確認
!nvidia-smi
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

### ステップ2: 公式推奨による依存関係インストール

```bash
# 1. 基本的な依存関係（公式推奨順）
!pip install diffusers>=0.19.0 transformers safetensors accelerate

# 2. SDXL専用ライブラリ
!pip install invisible_watermark

# 3. PyTorch（CUDA対応）
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. オプション：高速化ライブラリ
!pip install xformers
```

### ステップ3: SDXL パイプラインの初期化（公式推奨）

```python
from diffusers import DiffusionPipeline
import torch

# 公式推奨セットアップ
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,      # メモリ効率化
    use_safetensors=True,           # セキュリティ向上
    variant="fp16"                  # 軽量版モデル使用
)

# Colab無料版用メモリ効率化
pipe.enable_model_cpu_offload()   # 公式推奨：低VRAMデバイス用
```

### ステップ4: 動作確認テスト

```python
# 基本的な画像生成テスト
prompt = "A beautiful landscape with mountains and a lake, digital art"

# 生成実行
images = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=20,
    guidance_scale=7.5
).images

# 結果表示と保存
images[0].show()
images[0].save("test_sdxl_output.png")
```

## 🔧 Colab無料版最適化設定

### メモリ効率化（必須）

```python
# 1. CPU オフロード（公式推奨）
pipe.enable_model_cpu_offload()

# 2. VAE タイリング（高解像度用）
pipe.vae.enable_tiling()

# 3. 注意機構の最適化（オプション）
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("xformers メモリ最適化: 有効")
except:
    print("xformers: 利用不可（スキップ）")

# 4. メモリクリア
import gc
torch.cuda.empty_cache()
gc.collect()
```

### 高速化設定（Pro版推奨）

```python
# torch.compile() で20-30%高速化（公式推奨）
try:
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    print("torch.compile 高速化: 有効")
except:
    print("torch.compile: サポートされていません")

# 追加の最適化
pipe.enable_attention_slicing()
```

## 📝 設定パラメータ詳細

### 基本パラメータ

```python
# SDXL Base 1.0 推奨設定
sdxl_config = {
    "height": 1024,              # 標準解像度
    "width": 1024,               # 標準解像度
    "num_inference_steps": 30,   # 品質重視: 30-50
    "guidance_scale": 7.5,       # プロンプト重視度
    "num_images_per_prompt": 1,  # Colab無料版: 1枚ずつ
}
```

### 用途別最適化設定

#### YouTube サムネイル用（16:9）
```python
youtube_config = {
    "height": 576,               # 1024 * 9/16
    "width": 1024,
    "num_inference_steps": 25,
    "guidance_scale": 8.0,
}
```

#### Note アイキャッチ用（1.91:1）
```python
note_config = {
    "height": 536,               # 1024 / 1.91
    "width": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
}
```

#### 正方形アイコン用（1:1）
```python
icon_config = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 35,
    "guidance_scale": 8.5,
}
```

## 🛠️ 高度な設定とカスタマイズ

### リファイナーモデルの併用（Pro版推奨）

```python
# リファイナーモデルの追加（品質向上）
from diffusers import DiffusionPipeline

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.enable_model_cpu_offload()

# 2段階生成（Base → Refiner）
def generate_with_refiner(prompt, num_inference_steps=40, high_noise_frac=0.8):
    # Base モデルで初期生成
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    
    # Refiner で品質向上
    image = refiner(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]
    
    return image
```

### カスタムスケジューラの使用

```python
from diffusers import DPMSolverMultistepScheduler

# より効率的なスケジューラに変更
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 少ないステップで高品質生成
fast_config = {
    "num_inference_steps": 15,   # DPMSolver で削減可能
    "guidance_scale": 7.5,
}
```

## 📊 パフォーマンス監視

### メモリ使用量の監視

```python
def monitor_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    import psutil
    ram_usage = psutil.virtual_memory().percent
    print(f"RAM Usage: {ram_usage:.1f}%")

# 生成前後でメモリ監視
monitor_memory()
# ... 画像生成 ...
monitor_memory()
```

### 生成速度の測定

```python
import time

def benchmark_generation(prompt, config, iterations=3):
    times = []
    for i in range(iterations):
        start_time = time.time()
        image = pipe(prompt=prompt, **config).images[0]
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"Iteration {i+1}: {end_time - start_time:.2f}s")
    
    avg_time = sum(times) / len(times)
    print(f"Average generation time: {avg_time:.2f}s")
    return avg_time
```

## 🔧 トラブルシューティング

### よくある問題と解決法

#### 1. CUDA Out of Memory エラー
```python
# 解決策1: より積極的なCPUオフロード
pipe.enable_sequential_cpu_offload()

# 解決策2: バッチサイズを1に制限
config["num_images_per_prompt"] = 1

# 解決策3: 解像度を下げる
config["height"] = 768
config["width"] = 768
```

#### 2. 生成速度が遅い
```python
# 解決策1: ステップ数を削減
config["num_inference_steps"] = 20

# 解決策2: DPMSolver使用
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 解決策3: torch.compile（対応環境のみ）
pipe.unet = torch.compile(pipe.unet)
```

#### 3. モデル読み込みエラー
```python
# 解決策1: キャッシュクリア
!rm -rf ~/.cache/huggingface/

# 解決策2: 手動ダウンロード
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    use_auth_token=True  # 必要に応じて
)
```

## 📁 自動セットアップスクリプト

### setup_sdxl.py の作成

```python
#!/usr/bin/env python3
"""
SDXL Base 1.0 自動セットアップスクリプト
Google Colab環境でのStable Diffusion XL環境構築
"""

import subprocess
import sys
import torch
from diffusers import DiffusionPipeline

def install_dependencies():
    """公式推奨の依存関係をインストール"""
    packages = [
        "diffusers>=0.19.0",
        "transformers",
        "safetensors", 
        "accelerate",
        "invisible_watermark",
        "xformers"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def setup_sdxl_pipeline():
    """SDXL パイプラインのセットアップ"""
    try:
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        
        # Colab最適化
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
            
        print("✅ SDXL pipeline setup completed")
        return pipe
        
    except Exception as e:
        print(f"❌ SDXL pipeline setup failed: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Starting SDXL Base 1.0 setup for Google Colab...")
    install_dependencies()
    pipe = setup_sdxl_pipeline()
    
    if pipe:
        print("🎉 Setup completed successfully!")
        # テスト生成
        test_image = pipe("A beautiful sunset, digital art").images[0]
        test_image.save("sdxl_test.png")
        print("📸 Test image saved as 'sdxl_test.png'")
    else:
        print("💥 Setup failed. Please check the error messages above.")
```

## ✅ セットアップ確認チェックリスト

- [ ] GPU利用が有効（nvidia-smi で確認）
- [ ] diffusers >= 0.19.0 がインストール済み
- [ ] SDXL Base 1.0 モデルが正常に読み込める
- [ ] CPU オフロードが有効化済み
- [ ] VAE タイリングが有効化済み
- [ ] テスト画像生成が正常に完了
- [ ] メモリ使用量が許容範囲内
- [ ] 生成時間が許容範囲内（60-120秒）

セットアップ完了後、[SDXL ワークフロー](sdxl-workflow.md)に進んでください。