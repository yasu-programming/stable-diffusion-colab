# トラブルシューティングスクリプト

各種エラーや問題の修正用スクリプト集

## 🚨 主要スクリプト

### modern_fix.py
**最も推奨される現代的修正スクリプト**
- 現代的で確実に動作するバージョンで統一
- NumPy 1.26.4、PyTorch 2.2.2、Diffusers 0.30.0など最新版
- 最新機能を使用可能

```python
!python scripts/modern_fix.py
```

### final_fix.py
**安定性重視の修正スクリプト**
- 全ての互換性問題を一括解決
- やや古めのバージョンで確実性を重視
- xformers問題を回避

```python
!python scripts/final_fix.py
```

## 🔧 個別修正スクリプト

### numpy_fix.py
NumPy 2.0互換性問題の修正
- NumPy 2.0 → 1.x にダウングレード
- `broadcast_to`エラーの解決

### quick_fix.py
setup.py実行後の問題修正
- Hugging Face Hub互換性修正
- `cached_download`エラーの解決

### manual_setup.py
手動セットアップ（最小構成）
- 必要最小限のパッケージのみインストール
- 確実に動作する構成

### setup_simple.py
簡易セットアップスクリプト
- setup.pyの軽量版
- バージョン競合を最小化

## 📋 その他

### ultimate_fix.py
**最も古い安定版での修正**
- NumPy 1.21.6、PyTorch 2.0.1など確実に動作する組み合わせ
- 最新機能は使えないが確実性を最優先

### emergency_fix.py
緊急修正用（全削除→再インストール）
- 全パッケージ削除後に互換バージョンを順次インストール

### setup_fix.py
Hugging Face Hub エラー専用修正

## 🚀 使用方法

1. **現代的バージョンで試す（推奨）**
   ```python
   !python scripts/modern_fix.py
   ```

2. **安定性重視で試す**
   ```python
   !python scripts/final_fix.py
   ```

3. **特定の問題がある場合**
   ```python
   !python scripts/numpy_fix.py    # NumPy問題
   !python scripts/quick_fix.py    # HF Hub問題
   ```

4. **どうしても動かない場合（最古安定版）**
   ```python
   !python scripts/ultimate_fix.py
   !python scripts/manual_setup.py
   ```