"""
ユーザーベース協調フィルタリングの実装
Open/Closed Principleを実装
"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
from ..interfaces.recommendation_interface import IRecommendationStrategy


class UserBasedCollaborativeFiltering(IRecommendationStrategy):
    """ユーザーベース協調フィルタリング（Single Responsibility Principle）"""
    
    def __init__(self, data_provider, max_users: int = 10000, max_anime: int = 5000):
        self.data_provider = data_provider
        self.max_users = max_users  # 最大ユーザー数
        self.max_anime = max_anime  # 最大アニメ数
        self.user_similarity_matrix = None
        self.rating_matrix = None
        self.user_mapping = None
        self.anime_mapping = None
    
    def _create_rating_matrix(self):
        """評価行列を作成（メモリ効率化版）"""
        rating_data = self.data_provider.load_rating_data()
        anime_data = self.data_provider.load_anime_data()
        
        # 評価数の多いアニメを選択
        anime_rating_counts = rating_data.groupby('anime_id').size()
        popular_anime_ids = anime_rating_counts.nlargest(self.max_anime).index.tolist()
        
        # 評価数の多いユーザーを選択
        user_rating_counts = rating_data.groupby('user_id').size()
        active_user_ids = user_rating_counts.nlargest(self.max_users).index.tolist()
        
        # フィルタリングされたデータ
        filtered_ratings = rating_data[
            (rating_data['anime_id'].isin(popular_anime_ids)) &
            (rating_data['user_id'].isin(active_user_ids))
        ]
        
        # マッピングを作成
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(active_user_ids)}
        self.anime_mapping = {anime_id: idx for idx, anime_id in enumerate(popular_anime_ids)}
        
        # スパース行列を作成
        rows = [self.user_mapping[user_id] for user_id in filtered_ratings['user_id']]
        cols = [self.anime_mapping[anime_id] for anime_id in filtered_ratings['anime_id']]
        values = filtered_ratings['rating'].values
        
        self.rating_matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(len(active_user_ids), len(popular_anime_ids))
        )
        
        print(f"評価行列を作成しました: {self.rating_matrix.shape}")
    
    def _compute_user_similarity(self):
        """ユーザー類似度を計算"""
        if self.rating_matrix is None:
            self._create_rating_matrix()
        
        # コサイン類似度を計算（スパース行列版）
        self.user_similarity_matrix = cosine_similarity(self.rating_matrix)
        print(f"ユーザー類似度行列を作成しました: {self.user_similarity_matrix.shape}")
    
    def get_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """
        ユーザーに対するレコメンドを取得
        
        Args:
            user_id: ユーザーID
            n_recommendations: レコメンド数
            
        Returns:
            レコメンドリスト
        """
        if self.user_similarity_matrix is None:
            self._compute_user_similarity()
        
        # ユーザーがマッピングに含まれているかチェック
        if user_id not in self.user_mapping:
            return self._get_popular_recommendations(n_recommendations)
        
        user_idx = self.user_mapping[user_id]
        
        # 類似ユーザーを取得
        user_similarities = self.user_similarity_matrix[user_idx]
        similar_users = np.argsort(user_similarities)[::-1][1:11]  # 上位10ユーザー
        
        # 推薦スコアを計算
        anime_scores = {}
        user_ratings = self.rating_matrix[user_idx].toarray().flatten()
        
        for similar_user_idx in similar_users:
            similarity = user_similarities[similar_user_idx]
            similar_user_ratings = self.rating_matrix[similar_user_idx].toarray().flatten()
            
            # 類似ユーザーが評価したが、対象ユーザーが未評価のアニメを推薦
            for anime_idx, rating in enumerate(similar_user_ratings):
                if rating > 0 and user_ratings[anime_idx] == 0:
                    anime_id = list(self.anime_mapping.keys())[list(self.anime_mapping.values()).index(anime_idx)]
                    if anime_id not in anime_scores:
                        anime_scores[anime_id] = 0
                    anime_scores[anime_id] += similarity * rating
        
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
                    'recommendation_type': 'collaborative'
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