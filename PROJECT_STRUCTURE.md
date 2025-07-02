# プロジェクト構成

## 📁 ディレクトリ構造

```
stable-diffusion-colab/
├── README.md                    # メインの説明とクイックスタート
├── CLAUDE.md                    # Claude Code用ガイダンス
├── setup.py                     # メインセットアップスクリプト
├── requirements.txt             # 依存関係定義
├── colab_quickstart.ipynb       # Colabテンプレートノートブック
├── PROJECT_STRUCTURE.md         # このファイル
│
├── docs/                        # 詳細ドキュメント
│   ├── README.md               # ドキュメント一覧
│   ├── requirements.md         # プロジェクト要件
│   ├── colab-setup.md          # Colab初期設定
│   ├── colab-workflow.md       # 毎回のワークフロー
│   ├── model-recommendations.md # 商用利用可能モデル
│   └── usage-examples.md       # サンプルプロンプト集
│
├── scripts/                     # トラブルシューティング
│   ├── README.md               # スクリプト説明
│   ├── final_fix.py            # 最終修正（推奨）
│   ├── numpy_fix.py            # NumPy問題修正
│   ├── quick_fix.py            # HF Hub問題修正
│   ├── manual_setup.py         # 手動セットアップ
│   ├── emergency_fix.py        # 緊急修正
│   ├── setup_simple.py         # 簡易セットアップ
│   └── setup_fix.py            # HF Hub専用修正
│
└── (実行時に作成される)
    ├── generated_images/        # 生成画像保存先
    ├── models_cache/            # モデルキャッシュ
    ├── temp/                    # 一時ファイル
    └── quick_start.py           # 自動生成テストスクリプト
```

## 🚀 主要ファイル

### setup.py
メインのセットアップスクリプト
- GPU確認
- 互換性を考慮したパッケージインストール
- モデルダウンロード
- ディレクトリ作成

### colab_quickstart.ipynb
Google Colabで使用するテンプレートノートブック
- ワンクリック実行可能
- セットアップから画像生成まで完結

## 📚 ドキュメント

### docs/
詳細な説明とガイド
- **requirements.md**: プロジェクト要件定義
- **colab-workflow.md**: 毎回のクローン→生成ワークフロー
- **model-recommendations.md**: 商用利用可能モデル一覧
- **usage-examples.md**: YouTube・Note用サンプルプロンプト

## 🛠️ トラブルシューティング

### scripts/
各種エラー修正用スクリプト集
- **final_fix.py**: 最も推奨される一括修正
- **numpy_fix.py**: NumPy 2.0互換性問題
- **quick_fix.py**: Hugging Face Hub問題
- **manual_setup.py**: 最小構成セットアップ

## 🎯 使用の流れ

1. **初回**: README.mdの3ステップ手順
2. **エラー時**: scripts/final_fix.py
3. **カスタマイズ**: docs/内のガイド参照
4. **毎回**: colab_quickstart.ipynb使用

この構成により、初心者から上級者まで段階的に使用できます。