# トラブルシューティングスクリプト

各種エラーや問題の修正用スクリプト集

## 🚨 主要スクリプト

### final_fix.py
**最も推奨される修正スクリプト**
- 全ての互換性問題を一括解決
- NumPy、PyTorch、Diffusersの依存関係を正しい順序でインストール
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

### emergency_fix.py
緊急修正用（全削除→再インストール）
- 全パッケージ削除後に互換バージョンを順次インストール

### setup_fix.py
Hugging Face Hub エラー専用修正

## 🚀 使用方法

1. **まず最初に試す**
   ```python
   !python scripts/final_fix.py
   ```

2. **特定の問題がある場合**
   ```python
   !python scripts/numpy_fix.py    # NumPy問題
   !python scripts/quick_fix.py    # HF Hub問題
   ```

3. **どうしても動かない場合**
   ```python
   !python scripts/emergency_fix.py
   !python scripts/manual_setup.py
   ```