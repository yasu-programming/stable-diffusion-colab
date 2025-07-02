#!/usr/bin/env python3
"""
現代的な修正スクリプト - 適切に新しいバージョンで統一
"""

import subprocess
import sys
import os

def run_command(command, description=""):
    print(f"🔄 {description}")
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True)
        print(f"✅ {description} - 完了")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - 失敗")
        return False

def main():
    print("🚀 現代的な修正: 適切に新しいバージョンで統一")
    print("=" * 60)
    
    # 1. 完全削除
    print("🗑️ 全パッケージ削除...")
    packages = [
        "torch", "torchvision", "torchaudio", "numpy", 
        "diffusers", "transformers", "huggingface_hub", 
        "xformers", "accelerate", "safetensors"
    ]
    for package in packages:
        run_command(f"pip uninstall -y {package}", f"{package} 削除")
    
    # 2. 現代的で互換性のあるバージョンでインストール
    print("📦 現代的バージョンでインストール...")
    
    # NumPy 1.26.4 (Python 3.11対応、現代的だが安定)
    run_command("pip install numpy==1.26.4", "NumPy 1.26.4")
    
    # PyTorch 2.2.2 (NumPy 1.26.4と互換、比較的新しい)
    run_command(
        "pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118",
        "PyTorch 2.2.2"
    )
    
    # Hugging Face Hub 最新安定版
    run_command("pip install huggingface_hub==0.24.6", "Hugging Face Hub 0.24.6")
    
    # Diffusers 最新版 (NumPy 1.26.4と互換)
    run_command("pip install diffusers==0.30.0", "Diffusers 0.30.0")
    
    # Transformers 最新版
    run_command("pip install transformers==4.44.0", "Transformers 4.44.0")
    
    # その他現代的バージョン
    run_command("pip install accelerate==0.34.0", "Accelerate 0.34.0")
    run_command("pip install safetensors==0.4.4", "Safetensors 0.4.4")
    run_command("pip install pillow matplotlib", "画像処理ライブラリ")
    
    # xformers 現代版（オプション）
    print("🔧 xformers 現代版インストール試行...")
    xformers_success = run_command("pip install xformers==0.0.27", "xformers 0.0.27")
    if not xformers_success:
        print("⚠️ xformersインストール失敗 - スキップして続行")
    
    # 3. ディレクトリ作成
    os.makedirs("generated_images", exist_ok=True)
    os.makedirs("models_cache", exist_ok=True)
    print("✅ ディレクトリ作成完了")
    
    # 4. 現代的テストコード作成
    test_code = '''#!/usr/bin/env python3
"""
現代的テストコード - 最新機能対応
"""
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

print("🚀 現代的 Stable Diffusion テスト")
print("=" * 40)

# バージョン確認
import numpy as np
print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

# 現代的パイプライン初期化
print("📥 モデル読み込み中...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")
    
    # 現代的メモリ最適化
    try:
        pipe.enable_memory_efficient_attention()
        print("✅ メモリ効率化 ON")
    except:
        print("⚠️ メモリ効率化スキップ")
    
    try:
        pipe.enable_model_cpu_offload()
        print("✅ CPU Offload ON")
    except:
        print("⚠️ CPU Offloadスキップ")

print("🎨 高品質画像生成中...")
prompt = "a futuristic city skyline at sunset, cyberpunk style, highly detailed, 8k"
negative_prompt = "blurry, low quality, distorted"

# 高品質設定
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,  # 高品質
    guidance_scale=8.0,
    width=768,
    height=768
).images[0]

# 保存
image.save("generated_images/modern_test_hq.png")
print("✅ 高品質画像生成完了!")
print("💾 保存先: generated_images/modern_test_hq.png")

# 表示
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis("off")
plt.title("Modern High Quality Test")
plt.show()

print("🎉 現代的テスト成功 - 最新機能で動作中!")
'''
    
    with open("modern_test.py", "w") as f:
        f.write(test_code)
    
    # 5. 動作確認
    print("🔍 動作確認...")
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"✅ CUDA: {torch.cuda.is_available()}")
        
        from diffusers import StableDiffusionPipeline
        print("✅ Diffusers インポート成功")
        
        # 新機能テスト
        print(f"✅ NumPy dtypes: {hasattr(np, 'dtypes')}")
        
        print("=" * 60)
        print("🎉 現代的修正完了!")
        print("📝 現代的テスト: python modern_test.py")
        print("💡 最新機能を使用した高品質生成が可能です")
        
    except Exception as e:
        print(f"❌ 動作確認エラー: {e}")
        print("💡 ultimate_fix.py を試してください")

if __name__ == "__main__":
    main()