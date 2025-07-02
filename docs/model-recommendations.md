# 商用利用可能なStable Diffusionモデル推奨リスト

## 基本モデル（商用利用可能）

### 1. Stable Diffusion v1.5
```python
model_id = "runwayml/stable-diffusion-v1-5"
```
- **ライセンス**: CreativeML Open RAIL-M（商用利用可能）
- **特徴**: 安定性が高く、Colab無料版で安定動作
- **用途**: 汎用的な画像生成、初心者におすすめ

### 2. Stable Diffusion v2.1
```python
model_id = "stabilityai/stable-diffusion-2-1"
```
- **ライセンス**: CreativeML Open RAIL++ M（商用利用可能）
- **特徴**: v1.5より高品質、768x768ネイティブ
- **用途**: 高品質画像、アスペクト比1:1のアイコン

### 3. Stable Diffusion XL Base 1.0
```python
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
```
- **ライセンス**: CreativeML Open RAIL++ M（商用利用可能）
- **特徴**: 最高品質、1024x1024ネイティブ
- **注意**: メモリ使用量大、Colab無料版では制限あり

## 特化モデル（商用利用可能）

### アニメ・イラスト系
```python
# Anything V4.0
model_id = "andite/anything-v4.0"
# ライセンス: CreativeML Open RAIL-M
# 特徴: アニメスタイル、キャラクター生成に強い
```

### リアリスティック系
```python
# Realistic Vision V5.0
model_id = "SG161222/Realistic_Vision_V5.0_noVAE"
# ライセンス: CreativeML Open RAIL-M  
# 特徴: 写実的な人物・風景画像
```

## YouTubeサムネイル特化設定

### 推奨モデル
1. **Stable Diffusion v1.5** - 汎用性とコスパ
2. **Anything V4.0** - アニメ調サムネイル

### アスペクト比設定
```python
# YouTube サムネイル (16:9)
width, height = 1280, 720  # 高解像度
width, height = 896, 512   # Colab最適化版
```

## アイコン生成特化設定

### 推奨モデル
1. **Stable Diffusion v2.1** - 正方形ネイティブ
2. **Stable Diffusion XL** - 最高品質（リソース許可時）

### アスペクト比設定
```python
# アイコン (1:1)
width, height = 512, 512   # 標準
width, height = 768, 768   # 高解像度
```

## ライセンス注意事項

### CreativeML Open RAIL-M/++
- ✅ 商用利用可能
- ✅ 修正・改変可能
- ✅ 再配布可能
- ❌ 有害コンテンツ生成禁止
- ❌ 個人の肖像権侵害禁止

### 使用時の推奨事項
1. 生成画像に人物が含まれる場合は実在人物でないことを確認
2. 著作権のあるキャラクター・ロゴの直接模倣は避ける
3. 商用利用時はライセンス表記を検討