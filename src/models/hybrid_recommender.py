"""
ハイブリッドレコメンドシステムの実装
Open/Closed Principleを実装
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from ..interfaces.recommendation_interface import IRecommendationStrategy
from .collaborative_filtering import UserBasedCollaborativeFiltering
from .content_based_filtering import ContentBasedFiltering


class HybridRecommender(IRecommendationStrategy):
    """ハイブリッドレコメンダー（Single Responsibility Principle）"""
    
    def __init__(self, data_provider, collaborative_weight: float = 0.6, content_weight: float = 0.4):
        self.data_provider = data_provider
        self.collaborative_weight = collaborative_weight
        self.content_weight = content_weight
        
        # 各レコメンダーを初期化
        self.collaborative_filtering = UserBasedCollaborativeFiltering(data_provider)
        self.content_based_filtering = ContentBasedFiltering(data_provider)
    
    def get_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """
        ハイブリッドレコメンドを取得
        
        Args:
            user_id: ユーザーID
            n_recommendations: レコメンド数
            
        Returns:
            レコメンドリスト
        """
        # 各手法でレコメンドを取得
        collaborative_recs = self.collaborative_filtering.get_recommendations(user_id, n_recommendations * 2)
        content_recs = self.content_based_filtering.get_recommendations(user_id, n_recommendations * 2)
        
        # ハイブリッドスコアを計算
        hybrid_recs = self._combine_recommendations(collaborative_recs, content_recs, n_recommendations)
        
        return hybrid_recs
    
    def _combine_recommendations(self, collaborative_recs: List[Dict], content_recs: List[Dict], n_recommendations: int) -> List[Dict[str, Any]]:
        """レコメンドを組み合わせてハイブリッドスコアを計算"""
        # アニメIDをキーとしたスコア辞書を作成
        anime_scores = {}
        
        # 協調フィルタリングのスコアを追加
        for rec in collaborative_recs:
            anime_id = rec['anime_id']
            if anime_id not in anime_scores:
                anime_scores[anime_id] = {
                    'collaborative_score': 0,
                    'content_score': 0,
                    'anime_info': rec
                }
            anime_scores[anime_id]['collaborative_score'] = rec.get('similarity_score', 0)
        
        # コンテンツベースフィルタリングのスコアを追加
        for rec in content_recs:
            anime_id = rec['anime_id']
            if anime_id not in anime_scores:
                anime_scores[anime_id] = {
                    'collaborative_score': 0,
                    'content_score': 0,
                    'anime_info': rec
                }
            anime_scores[anime_id]['content_score'] = rec.get('similarity_score', 0)
        
        # ハイブリッドスコアを計算
        hybrid_recommendations = []
        for anime_id, scores in anime_scores.items():
            hybrid_score = (
                self.collaborative_weight * scores['collaborative_score'] +
                self.content_weight * scores['content_score']
            )
            
            anime_info = scores['anime_info'].copy()
            anime_info['hybrid_score'] = hybrid_score
            anime_info['collaborative_score'] = scores['collaborative_score']
            anime_info['content_score'] = scores['content_score']
            anime_info['recommendation_type'] = 'hybrid'
            
            hybrid_recommendations.append(anime_info)
        
        # ハイブリッドスコアでソート
        hybrid_recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return hybrid_recommendations[:n_recommendations]
    
    def get_diverse_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """多様性を考慮したレコメンドを取得"""
        collaborative_recs = self.collaborative_filtering.get_recommendations(user_id, n_recommendations)
        content_recs = self.content_based_filtering.get_recommendations(user_id, n_recommendations)
        
        # 各手法から交互に選択して多様性を確保
        diverse_recs = []
        max_len = max(len(collaborative_recs), len(content_recs))
        
        for i in range(max_len):
            if i < len(collaborative_recs):
                rec = collaborative_recs[i].copy()
                rec['recommendation_type'] = 'hybrid_collaborative'
                diverse_recs.append(rec)
            
            if i < len(content_recs) and len(diverse_recs) < n_recommendations:
                rec = content_recs[i].copy()
                rec['recommendation_type'] = 'hybrid_content'
                diverse_recs.append(rec)
        
        return diverse_recs[:n_recommendations]
    
    def get_personalized_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """ユーザーの嗜好に基づいて重みを調整したレコメンドを取得"""
        # ユーザーの評価履歴を分析
        user_ratings = self.data_provider.get_user_ratings(user_id)
        
        if user_ratings.empty:
            return self.get_recommendations(user_id, n_recommendations)
        
        # ユーザーの評価パターンを分析
        rating_std = user_ratings['rating'].std()
        rating_mean = user_ratings['rating'].mean()
        
        # 評価の一貫性に基づいて重みを調整
        if rating_std < 1.0:  # 評価が一貫している場合
            # コンテンツベースフィルタリングを重視
            self.collaborative_weight = 0.3
            self.content_weight = 0.7
        elif rating_std > 2.0:  # 評価が多様な場合
            # 協調フィルタリングを重視
            self.collaborative_weight = 0.8
            self.content_weight = 0.2
        else:
            # デフォルトの重み
            self.collaborative_weight = 0.6
            self.content_weight = 0.4
        
        return self.get_recommendations(user_id, n_recommendations) 