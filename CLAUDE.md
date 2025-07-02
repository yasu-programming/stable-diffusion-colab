# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Google Colab無料版でStable Diffusion XL Base 1.0を使用する商用利用可能な画像生成システム。YouTube・Note用のサムネイル・アイコン生成に特化。

## Commands

### Main Setup
```bash
python setup.py  # SDXL自動セットアップ（推奨）
```

### Testing
```bash
python sdxl_test.py              # SDXL詳細テスト
```

### Troubleshooting
```bash
python scripts/sdxl_fix.py       # SDXL修正（推奨）
python scripts/memory_optimizer.py  # メモリ最適化
```

## Architecture

### Core Dependencies (SDXL Optimized)
- **diffusers >= 0.19.0**: SDXL対応版
- **transformers**: 最新版
- **safetensors**: セキュアなモデル読み込み
- **accelerate**: 高速化ライブラリ
- **invisible_watermark**: SDXL専用ライブラリ
- **PyTorch 2.0+**: CUDA対応（公式推奨）
- **xformers**: メモリ効率化（オプション）

### Directory Structure
```
stable-diffusion-colab/
├── setup.py                    # SDXL自動セットアップ
├── sdxl_test.py                # SDXL詳細テスト
├── sdxl_colab_quickstart.ipynb # Google Colabテンプレート
├── docs/                       # 詳細ドキュメント
│   ├── README.md              # プロジェクト概要
│   ├── sdxl-setup.md          # SDXL セットアップガイド
│   ├── sdxl-workflow.md       # SDXL ワークフロー
│   ├── sdxl-examples.md       # 使用例・サンプルコード
│   └── requirements.md        # システム要件
├── scripts/                    # トラブルシューティング
│   ├── sdxl_fix.py            # SDXL修正スクリプト
│   └── memory_optimizer.py    # メモリ最適化
├── generated_images/           # 生成画像（実行時作成）
└── logs/                      # テスト結果（実行時作成）
```

### SDXL Setup Strategy
1. **公式準拠**: Hugging Face公式ドキュメントに基づく実装
2. **段階的インストール**: diffusers → transformers → PyTorch → オプション
3. **メモリ最適化**: Google Colab無料版対応の最適化
4. **エラーハンドリング**: xformers等のオプション機能は失敗時スキップ

### Common Issues and Solutions (SDXL)
- **CUDA Out of Memory**: CPU オフロード・VAE タイリングで解決
- **モデル読み込み失敗**: safetensors・variant="fp16"で解決
- **生成速度遅い**: torch.compile・xformersで高速化
- **メモリ効率化**: 専用最適化スクリプトで自動調整