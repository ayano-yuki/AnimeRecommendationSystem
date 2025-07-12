# アニメレコメンドシステム

Kaggleの「Anime Recommendation Database 2020」データセットを使用したアニメレコメンドシステムです。

## 機能

- ユーザーベース協調フィルタリング
- コンテンツベースフィルタリング（ジャンル・シノプシス活用）
- ハイブリッドレコメンド
- Webインターフェース（Streamlit）
- 大規模データセット対応（57M+評価データ）

## セットアップ

### 1. UVプロジェクトの初期化

```bash
# 依存関係のインストール
uv sync

# 仮想環境の有効化
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # Linux/Mac
```

### 2. データセットの準備

Kaggleの「[Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020)」データセットを `data/` ディレクトリに配置：

```
data/
├── anime_with_synopsis.csv  # アニメ情報（16,214件）
├── rating_complete.csv      # 評価データ（57M+件）
├── animelist.csv           # ユーザーアニメリスト
└── watching_status.csv     # 視聴ステータス
```

### 3. アプリケーションの実行

```bash
streamlit run src/app/main.py
```

## プロジェクト構造

```
anime-recommendation-system/
├── src/
│   ├── data/           # データ処理
│   │   ├── download_dataset.py
│   │   └── data_provider.py
│   ├── models/         # レコメンドモデル
│   │   ├── collaborative_filtering.py
│   │   ├── content_based_filtering.py
│   │   └── hybrid_recommender.py
│   ├── services/       # ビジネスロジック
│   │   └── recommendation_service.py
│   ├── interfaces/     # インターフェース定義
│   │   └── recommendation_interface.py
│   └── app/           # Webアプリケーション
│       └── main.py
├── tests/             # テストコード
├── data/              # データセット
├── notebooks/         # Jupyterノートブック
├── pyproject.toml     # UVプロジェクト設定
└── requirements.txt   # 依存関係（従来）
```

## データセット情報

- **アニメ数**: 16,214件
- **評価数**: 57,633,278件
- **ユーザー数**: 310,059人
- **特徴**: ジャンル、シノプシス、評価スコアを含む

## レコメンド手法

### 1. ユーザーベース協調フィルタリング
- 類似ユーザーの評価パターンに基づく推薦
- コサイン類似度を使用
- メモリ効率化のためサンプリング対応

### 2. コンテンツベースフィルタリング
- アニメの特徴（ジャンル、シノプシス）に基づく推薦
- TF-IDFベクトライザーでテキスト特徴量を生成
- コサイン類似度で類似アニメを検索

### 3. ハイブリッドレコメンド
- 協調フィルタリングとコンテンツベースフィルタリングを組み合わせ
- ユーザーの評価パターンに応じて重みを動的調整
- 多様性を考慮した推薦も可能

## SOLID原則の実装

- **Single Responsibility Principle**: 各クラスが単一の責任を持つ
- **Open/Closed Principle**: 新しいレコメンド手法を追加可能
- **Liskov Substitution Principle**: インターフェースの実装
- **Interface Segregation Principle**: 必要最小限のインターフェース
- **Dependency Inversion Principle**: 抽象に依存

## パフォーマンス最適化

### メモリ効率化
- 大規模データセット（57M+評価）に対応
- スパース行列の使用
- データサンプリング機能

### 処理速度向上
- キャッシュ機能の実装
- 段階的データ処理
- 並列処理対応

## 開発環境

- **Python**: 3.9+
- **パッケージ管理**: UV
- **Webフレームワーク**: Streamlit
- **機械学習**: scikit-learn
- **データ処理**: pandas, numpy

## テスト

```bash
# データプロバイダーのテスト
python -m unittest tests/test_data_provider.py

# レコメンドサービスのテスト
python -m unittest tests/test_recommendation_service.py
```

## 今後の改善予定

- [ ] メモリ効率化の実装
- [ ] データサンプリング機能の追加
- [ ] リアルタイム推薦の実装
- [ ] ユーザーインターフェースの改善
- [ ] 推薦精度の評価機能

## 注意事項

- 大規模データセットのため、十分なメモリ（8GB+）が必要
- 初回起動時はデータ読み込みに時間がかかる場合があります
- 推薦生成時はメモリ使用量が増加します 
