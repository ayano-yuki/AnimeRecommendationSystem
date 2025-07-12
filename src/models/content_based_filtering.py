"""
コンテンツベースフィルタリングの実装
Open/Closed Principleを実装
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Optional
from ..interfaces.recommendation_interface import IRecommendationStrategy


class ContentBasedFiltering(IRecommendationStrategy):
    """コンテンツベースフィルタリング（Single Responsibility Principle）"""
    
    def __init__(self, data_provider):
        self.data_provider = data_provider
        self.anime_data = None
        self.similarity_matrix = None
        self.vectorizer = None
        self.scaler = StandardScaler()
    
    def _prepare_features(self):
        """アニメの特徴量を準備"""
        anime_data = self.data_provider.load_anime_data()
        
        # ジャンルとタイプを結合してテキスト特徴量を作成
        anime_data['features'] = anime_data.apply(
            lambda row: f"{row.get('genre', '')} {row.get('type', '')} {row.get('name', '')}", 
            axis=1
        )
        
        # 数値特徴量の準備
        numeric_features = ['episodes', 'rating', 'members']
        for feature in numeric_features:
            if feature in anime_data.columns:
                anime_data[feature] = pd.to_numeric(anime_data[feature], errors='coerce').fillna(0)
        
        self.anime_data = anime_data
        return anime_data
    
    def _create_text_features(self):
        """テキスト特徴量を作成"""
        if self.anime_data is None:
            self._prepare_features()
        
        # TF-IDFベクトライザーでテキスト特徴量を作成
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        text_features = self.vectorizer.fit_transform(self.anime_data['features'].fillna(''))
        return text_features
    
    def _create_numeric_features(self):
        """数値特徴量を作成"""
        numeric_features = ['episodes', 'rating', 'members']
        available_features = [f for f in numeric_features if f in self.anime_data.columns]
        
        if available_features:
            numeric_data = self.anime_data[available_features].fillna(0)
            scaled_numeric = self.scaler.fit_transform(numeric_data)
            return scaled_numeric
        return np.zeros((len(self.anime_data), 1))
    
    def build_similarity_matrix(self):
        """類似度行列を構築"""
        if self.anime_data is None:
            self._prepare_features()
        
        # テキスト特徴量
        text_features = self._create_text_features()
        
        # 数値特徴量
        numeric_features = self._create_numeric_features()
        
        # 特徴量を結合
        if numeric_features.shape[1] > 0:
            combined_features = np.hstack([text_features.toarray(), numeric_features])
        else:
            combined_features = text_features.toarray()
        
        # コサイン類似度を計算
        self.similarity_matrix = cosine_similarity(combined_features)
        
        print(f"類似度行列を構築しました: {self.similarity_matrix.shape}")
    
    def get_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """
        ユーザーに対するレコメンドを取得
        
        Args:
            user_id: ユーザーID
            n_recommendations: レコメンド数
            
        Returns:
            レコメンドリスト
        """
        if self.similarity_matrix is None:
            self.build_similarity_matrix()
        
        # ユーザーの評価履歴を取得
        user_ratings = self.data_provider.get_user_ratings(user_id)
        
        if user_ratings.empty:
            # 評価履歴がない場合は人気アニメを推薦
            return self._get_popular_recommendations(n_recommendations)
        
        # ユーザーが高評価したアニメを取得
        high_rated_anime = user_ratings[user_ratings['rating'] >= 7]['anime_id'].tolist()
        
        if not high_rated_anime:
            return self._get_popular_recommendations(n_recommendations)
        
        # 類似度に基づいてレコメンドを生成
        recommendations = self._get_similar_anime(high_rated_anime, user_ratings, n_recommendations)
        
        return recommendations
    
    def _get_similar_anime(self, liked_anime_ids: List[int], user_ratings: pd.DataFrame, n_recommendations: int) -> List[Dict[str, Any]]:
        """類似アニメを取得"""
        # ユーザーが既に評価したアニメを除外
        rated_anime_ids = set(user_ratings['anime_id'].tolist())
        
        # 類似度スコアを計算
        anime_scores = {}
        
        for anime_id in liked_anime_ids:
            if anime_id in self.anime_data['anime_id'].values:
                anime_idx = self.anime_data[self.anime_data['anime_id'] == anime_id].index[0]
                similarities = self.similarity_matrix[anime_idx]
                
                for idx, similarity in enumerate(similarities):
                    candidate_anime_id = self.anime_data.iloc[idx]['anime_id']
                    
                    if candidate_anime_id not in rated_anime_ids:
                        if candidate_anime_id not in anime_scores:
                            anime_scores[candidate_anime_id] = 0
                        anime_scores[candidate_anime_id] += similarity
        
        # スコアでソートしてレコメンドを生成
        sorted_anime = sorted(anime_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for anime_id, score in sorted_anime[:n_recommendations]:
            anime_info = self.data_provider.get_anime_info(anime_id)
            if anime_info:
                recommendations.append({
                    'anime_id': anime_id,
                    'name': anime_info.get('name', 'Unknown'),
                    'genre': anime_info.get('genre', ''),
                    'type': anime_info.get('type', ''),
                    'episodes': anime_info.get('episodes', 0),
                    'rating': anime_info.get('rating', 0),
                    'similarity_score': score,
                    'recommendation_type': 'content_based'
                })
        
        return recommendations
    
    def _get_popular_recommendations(self, n_recommendations: int) -> List[Dict[str, Any]]:
        """人気アニメを推薦"""
        popular_anime = self.data_provider.get_popular_anime(min_ratings=50)
        
        recommendations = []
        for _, anime in popular_anime.head(n_recommendations).iterrows():
            recommendations.append({
                'anime_id': anime['anime_id'],
                'name': anime.get('name', 'Unknown'),
                'genre': anime.get('genre', ''),
                'type': anime.get('type', ''),
                'episodes': anime.get('episodes', 0),
                'rating': anime.get('rating', 0),
                'similarity_score': 0.0,
                'recommendation_type': 'popular'
            })
        
        return recommendations 