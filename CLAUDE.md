# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Google Colab無料版で動作する商用利用可能なStable Diffusion画像生成システム。YouTube・Note用のサムネイル・アイコン生成に特化。

## Commands

### Main Setup
```bash
python setup.py  # 自動セットアップ（推奨）
```

### Troubleshooting
```bash
python scripts/final_fix.py      # 最終修正（確実）
python scripts/modern_fix.py     # 現代的修正
python scripts/ultimate_fix.py   # 古い安定版修正
```

### Testing
```bash
python quick_start.py            # 基本テスト
python modern_test.py            # 現代的テスト
python ultra_stable_test.py      # 安定版テスト
```

## Architecture

### Core Dependencies
- **NumPy 1.26.4**: 現代的で安定したバージョン
- **PyTorch 2.2.2**: CUDA 11.8対応、NumPy 1.26.4と互換
- **Diffusers 0.30.0**: 最新機能対応
- **Transformers 4.44.0**: 最新版
- **xformers 0.0.27**: メモリ効率化（オプション）

### Directory Structure
```
stable-diffusion-colab/
├── setup.py                 # メインセットアップ
├── docs/                    # 詳細ドキュメント
├── scripts/                 # トラブルシューティング
├── generated_images/        # 生成画像（実行時作成）
├── models_cache/           # モデルキャッシュ（実行時作成）
└── colab_quickstart.ipynb  # Colabテンプレート
```

### Version Compatibility Strategy
1. **完全削除**: 既存の問題パッケージを削除
2. **順序インストール**: NumPy → PyTorch → HF Hub → Diffusers
3. **バージョン固定**: 互換性確認済みバージョンに固定
4. **エラーハンドリング**: xformers等のオプション機能は失敗時スキップ

### Common Issues and Solutions
- **NumPy 2.0問題**: NumPy 1.26.4に固定で解決
- **cached_download問題**: HF Hub 0.24.6で解決
- **xformers互換性**: 0.0.27使用、失敗時はスキップ
- **メモリ効率化**: エラーハンドリング付きで安全に有効化