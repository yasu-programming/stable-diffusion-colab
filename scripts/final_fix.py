#!/usr/bin/env python3
"""
最終修正スクリプト - 全ての互換性問題を解決
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
    except subprocess.CalledProcessError:
        print(f"❌ {description} - 失敗")
        return False

def main():
    print("🔧 最終修正: 全互換性問題解決")
    print("=" * 50)
    
    # 1. 全削除
    print("🗑️ 問題パッケージ完全削除...")
    packages = ["torch", "torchvision", "torchaudio", "numpy", "diffusers", "transformers", "huggingface_hub", "xformers", "accelerate"]
    for package in packages:
        run_command(f"pip uninstall -y {package}", f"{package} 削除")
    
    # 2. 順序正しくインストール
    print("📦 互換パッケージ順次インストール...")
    
    # NumPy 1.24.3 (確実に動作)
    run_command("pip install numpy==1.24.3", "NumPy 1.24.3")
    
    # PyTorch 2.1.0 (NumPy 1.24.3と互換)
    run_command(
        "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118",
        "PyTorch 2.1.0"
    )
    
    # Hugging Face Hub 0.20.3 (cached_download問題解決)
    run_command("pip install huggingface_hub==0.20.3", "Hugging Face Hub 0.20.3")
    
    # Diffusers & Transformers
    run_command("pip install diffusers==0.25.1", "Diffusers 0.25.1")
    run_command("pip install transformers==4.36.0", "Transformers 4.36.0")
    
    # その他
    run_command("pip install accelerate==0.25.0", "Accelerate")
    run_command("pip install safetensors==0.4.0", "Safetensors")
    run_command("pip install pillow matplotlib", "画像処理ライブラリ")
    
    # xformersはスキップ（問題の元）
    print("⚠️ xformersはスキップ（互換性問題のため）")
    
    # 3. ディレクトリ作成
    os.makedirs("generated_images", exist_ok=True)
    os.makedirs("models_cache", exist_ok=True)
    print("✅ ディレクトリ作成完了")
    
    # 4. 簡単なテストコード作成
    test_code = '''#!/usr/bin/env python3
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

print("🚀 Stable Diffusion テスト開始")

# パイプライン初期化
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")
    print("✅ GPU使用")
else:
    print("⚠️ CPU使用（遅い）")

# 画像生成
prompt = "a beautiful sunset over mountains, professional photography"
print(f"🎨 生成中: {prompt}")

image = pipe(
    prompt=prompt,
    num_inference_steps=20,
    guidance_scale=7.5,
    width=512,
    height=512
).images[0]

# 保存と表示
image.save("generated_images/test_final.png")
print("✅ 画像生成完了: generated_images/test_final.png")

# 表示
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis("off")
plt.title("Generated Image")
plt.show()
'''
    
    with open("test_final.py", "w") as f:
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
        
        print("=" * 50)
        print("🎉 最終修正完了！")
        print("📝 テスト実行: python test_final.py")
        
    except Exception as e:
        print(f"❌ 動作確認エラー: {e}")

if __name__ == "__main__":
    main()