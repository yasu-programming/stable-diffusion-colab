# Google Colab SDXL 画像生成システム

YouTube・Note用のサムネイル・アイコン生成に特化した、Google Colab無料版でStable Diffusion XL Base 1.0を使用する商用利用可能な画像生成システムです。

## 🎯 主要な特徴

- **SDXL Base 1.0使用**: 最新の高品質画像生成モデル
- **Google Colab無料版対応**: メモリ効率化による最適化
- **商用利用可能**: CreativeML Open RAIL++-M ライセンス
- **サムネイル・アイコン特化**: YouTube・Note等のコンテンツ制作に最適化
- **公式準拠**: Hugging Face公式ドキュメントに基づく実装

## 📋 システム要件

### 対応環境
- Google Colab（無料版・Pro版）
- Python 3.10+
- CUDA対応GPU（T4以上推奨）

### SDXL Base 1.0 要件
- **最小VRAM**: 8GB
- **推奨VRAM**: 12GB以上
- **RAM**: 12GB以上
- **解像度**: 1024×1024（標準）

## 🏗️ 技術スタック

### 核となる依存関係（公式推奨）
- **diffusers**: >= 0.19.0（SDXL対応版）
- **transformers**: 最新版
- **safetensors**: セキュアなモデル読み込み
- **accelerate**: 高速化ライブラリ
- **invisible_watermark**: 透かし機能
- **PyTorch**: 2.0+ (CUDA対応)

### 使用モデル
- **Stable Diffusion XL Base 1.0** (stabilityai/stable-diffusion-xl-base-1.0)
  - ライセンス: CreativeML Open RAIL++-M
  - 商用利用: 可能（研究・アート生成・教育用途）
  - 解像度: 1024×1024（標準）

## 🚀 クイックスタート

### 1. セットアップ手順

#### Google Colab での開始
1. **GPU有効化**: `ランタイム` → `ランタイムのタイプを変更` → `GPU`
2. **ノートブック使用**: `sdxl_colab_quickstart.ipynb` を開いて実行

#### Google Colab セル実行での開始
```bash
# リポジトリクローン
!git clone https://github.com/yasu-programming/stable-diffusion-colab.git
%cd stable-diffusion-colab

# 自動セットアップ（SDXL対応）
!python setup.py

# 動作確認テスト
!python sdxl_test.py
```

#### ローカル環境での開始
```bash
# リポジトリクローン
git clone https://github.com/yasu-programming/stable-diffusion-colab.git
cd stable-diffusion-colab

# 自動セットアップ（SDXL対応）
python setup.py

# 動作確認テスト
python sdxl_test.py
```

### 2. 画像生成手順

#### YouTube サムネイル生成
```python
from diffusers import DiffusionPipeline
import torch

# パイプライン初期化
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
pipe.enable_model_cpu_offload()

# YouTube サムネイル生成（16:9）
images = pipe(
    prompt="Excited gamer with headphones playing video game, vibrant colors, YouTube thumbnail style",
    height=576, width=1024,
    num_inference_steps=25,
    guidance_scale=8.0
).images

# 保存
images[0].save("youtube_thumbnail.png")
```

#### Note・ブログ画像生成
```python
# Note 記事用画像生成（1.91:1）
blog_images = pipe(
    prompt="Modern laptop on clean desk with coffee, minimalist workspace, professional photography",
    height=536, width=1024,
    num_inference_steps=30,
    guidance_scale=7.5
).images

blog_images[0].save("blog_header.png")
```

#### アイコン・ロゴ生成
```python
# アイコン生成（1:1）
icon_images = pipe(
    prompt="Simple geometric logo design, modern minimalist style, clean lines",
    height=1024, width=1024,
    num_inference_steps=35,
    guidance_scale=9.0
).images

icon_images[0].save("brand_icon.png")
```

### 3. 生成された画像の確認
生成された画像は `generated_images/` フォルダに保存されます。

## 📚 詳細ドキュメント

- [SDXL セットアップガイド](sdxl-setup.md) - SDXL専用セットアップ手順
- [SDXL ワークフロー](sdxl-workflow.md) - SDXL使用方法とベストプラクティス
- [SDXL 使用例](sdxl-examples.md) - サムネイル・アイコン生成の具体例
- [システム要件](requirements.md) - 詳細な技術要件と制限事項

## 🔧 SDXL専用機能

### メモリ効率化
- **CPU オフロード**: 無料版での動作を可能に
- **VAE タイリング**: 高解像度生成時のメモリ節約
- **FP16精度**: VRAM使用量削減

### 高速化オプション
- **torch.compile()**: 20-30%の速度向上
- **xformers**: 注意機構の最適化
- **バッチ処理**: 効率的な複数画像生成

## 🎨 主な用途と適用例

### YouTube サムネイル
- **解像度**: 1280×720（16:9）
- **特徴**: 鮮やかな色彩、明確な構図
- **生成時間**: 60-120秒/枚

### Note アイキャッチ
- **解像度**: 1200×630（1.91:1）
- **特徴**: 洗練されたデザイン、読みやすさ重視
- **生成時間**: 60-120秒/枚

### アイコン・ロゴ
- **解像度**: 1024×1024（1:1）
- **特徴**: シンプル、記憶に残るデザイン
- **生成時間**: 45-90秒/枚

## ⚖️ ライセンスと商用利用

### CreativeML Open RAIL++-M ライセンス
- **商用利用**: 可能
- **画像加工**: 自由に編集可能
- **再配布**: 生成画像の配布可能
- **用途制限**: 違法・有害コンテンツは禁止

### 推奨用途
- アート生成
- 教育ツール
- 研究用途
- コンテンツ制作

## 🔧 トラブルシューティング

### Colab無料版での制限対処
```bash
# メモリ不足時の対処
python scripts/sdxl_memory_fix.py

# 高速化設定
python scripts/sdxl_optimize.py
```

### よくある問題
1. **CUDA Out of Memory** → CPU オフロード有効化
2. **生成速度が遅い** → torch.compile() 使用
3. **品質が低い** → 推論ステップ数増加

## 📊 パフォーマンス目安

### Google Colab T4 GPU
- **生成時間**: 60-120秒/枚
- **最大解像度**: 1024×1024
- **連続生成**: 10-20枚程度

### 最適化後の性能
- **torch.compile使用**: 40-80秒/枚
- **メモリ効率化**: 6GB VRAM で動作可能
- **バッチ処理**: 効率向上

## 🎯 推奨ワークフロー

1. **プロンプト設計**: 目的に応じたプロンプト作成
2. **設定調整**: 解像度・ステップ数の最適化
3. **バッチ生成**: 複数バリエーション作成
4. **後処理**: 必要に応じた画像編集
5. **形式変換**: 用途に応じたフォーマット調整

## 📞 サポート

問題が発生した場合：
1. [SDXL セットアップガイド](sdxl-setup.md)で環境確認
2. [トラブルシューティング](#🔧-トラブルシューティング)を実行
3. [SDXL 使用例](sdxl-examples.md)でサンプル確認