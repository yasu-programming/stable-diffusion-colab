# stable-diffusion-colab

Google Colab無料版で動作する商用利用可能なStable Diffusion画像生成システム

## 🚀 クイックスタート（3ステップ）

### 1. Google Colab でGPU設定
- [Google Colab](https://colab.research.google.com/) にアクセス
- **ランタイム** → **ランタイムのタイプを変更** → **GPU (T4)** を選択

### 2. リポジトリクローン & セットアップ
```python
!git clone https://github.com/YOUR_USERNAME/stable-diffusion-colab.git
%cd stable-diffusion-colab
!python setup.py
```

### 3. 画像生成開始
```python
!python quick_start.py
```

## 📁 主な用途

- **YouTubeサムネイル**生成 (16:9)
- **Noteアイキャッチ画像**生成
- **アイコン**生成 (1:1)
- その他SNS用画像素材

## 📚 詳細ドキュメント

詳細な使用方法は [docs/](./docs/) フォルダをご参照ください：

- [colab-workflow.md](./docs/colab-workflow.md) - 毎回のワークフロー
- [model-recommendations.md](./docs/model-recommendations.md) - 商用利用可能モデル
- [usage-examples.md](./docs/usage-examples.md) - サンプルプロンプト集

## ⚖️ ライセンス

商用利用可能なCreativeML Open RAILライセンスのモデルを使用
