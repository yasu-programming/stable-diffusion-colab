#!/usr/bin/env python3
"""
手動セットアップ用 - 最小構成で確実に動作
"""

import subprocess
import os

def install_package(package, description):
    print(f"📦 {description}...")
    try:
        subprocess.run(f"pip install {package}", shell=True, check=True)
        print(f"✅ {description} 完了")
        return True
    except:
        print(f"❌ {description} 失敗")
        return False

def main():
    print("🔧 手動セットアップ（最小構成）")
    print("=" * 50)
    
    # ディレクトリ作成
    os.makedirs("generated_images", exist_ok=True)
    os.makedirs("models_cache", exist_ok=True)
    print("✅ ディレクトリ作成完了")
    
    # 必要最小限のパッケージ
    packages = [
        ('numpy==1.24.3', 'NumPy 1.24.3'),
        ('torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118', 'PyTorch 2.1.0'),
        ('huggingface_hub==0.20.3', 'Hugging Face Hub'),
        ('diffusers==0.25.1', 'Diffusers'),
        ('transformers==4.36.0', 'Transformers'),
        ('accelerate', 'Accelerate'),
        ('safetensors', 'Safetensors'),
        ('pillow', 'Pillow')
    ]
    
    success_count = 0
    for package, desc in packages:
        if install_package(package, desc):
            success_count += 1
    
    print("=" * 50)
    print(f"📊 結果: {success_count}/{len(packages)} パッケージ成功")
    
    # 動作確認
    try:
        import torch
        import numpy as np
        from diffusers import StableDiffusionPipeline
        
        print("✅ 基本動作確認OK")
        print(f"PyTorch: {torch.__version__}")
        print(f"NumPy: {np.__version__}")
        print(f"CUDA: {torch.cuda.is_available()}")
        
        # 簡単なテスト用コード生成
        test_code = '''
# 簡単テスト用
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")

# 画像生成テスト
prompt = "a beautiful sunset over mountains"
image = pipe(prompt, num_inference_steps=10).images[0]
image.save("generated_images/test.png")
print("✅ テスト画像生成完了: generated_images/test.png")
'''
        
        with open("simple_test.py", "w") as f:
            f.write(test_code)
        
        print("✅ simple_test.py 作成完了")
        print("📝 テスト実行: python simple_test.py")
        
    except Exception as e:
        print(f"❌ 動作確認エラー: {e}")

if __name__ == "__main__":
    main()