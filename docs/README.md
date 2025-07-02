# ドキュメント一覧

Google Colab無料版でStable Diffusionを使用した商用利用可能な画像生成システムのドキュメント集です。

## 📋 ドキュメント構成

### [requirements.md](./requirements.md)
プロジェクトの要件定義と仕様
- 実行環境（Google Colab無料版）
- ライセンス要件（商用利用可能）
- 使用用途（YouTube、Note等）
- 技術要件と出力仕様

### [colab-setup.md](./colab-setup.md) 
Google Colabでのセットアップ手順
- GPU設定方法
- 必要ライブラリのインストール
- モデルダウンロード手順
- メモリ最適化設定

### [model-recommendations.md](./model-recommendations.md)
商用利用可能なモデル推奨リスト
- 基本モデル（SD v1.5, v2.1, XL）
- 特化モデル（アニメ、リアル系）
- 用途別推奨設定
- ライセンス注意事項

### [usage-examples.md](./usage-examples.md)
具体的な使用例とサンプルプロンプト
- YouTubeサムネイル生成例
- アイコン生成例
- Note記事アイキャッチ例
- プロンプト作成のコツ

### [colab-workflow.md](./colab-workflow.md)
毎回のGitHubクローンワークフロー
- 3ステップクイックスタート
- 効率的なセットアップ手順
- ファイル管理とダウンロード方法
- トラブルシューティング

## 🚀 クイックスタート

### 毎回のワークフロー（推奨）
1. [colab-workflow.md](./colab-workflow.md) - GitHubクローン→即座に画像生成
2. `colab_quickstart.ipynb` をGoogle Colabで開いてワンクリック実行

### 初回セットアップ
1. [colab-setup.md](./colab-setup.md) でGoogle Colabを設定
2. [model-recommendations.md](./model-recommendations.md) で適切なモデルを選択
3. [usage-examples.md](./usage-examples.md) のサンプルコードを参考に画像生成

## ⚖️ ライセンスについて

本プロジェクトで推奨するモデルは全て**商用利用可能**ですが、生成時は以下に注意してください：
- 実在人物の肖像権
- 著作権のあるキャラクター・ロゴの模倣
- 有害コンテンツの生成禁止