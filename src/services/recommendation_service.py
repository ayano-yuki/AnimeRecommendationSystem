"""
レコメンドサービスの実装
Dependency Inversion Principleを実装
"""
import pandas as pd
from typing import List, Dict, Any, Optional
from ..interfaces.recommendation_interface import IRecommendationService, IRecommendationStrategy
from ..data.data_provider import AnimeDataProvider
from ..models.collaborative_filtering import UserBasedCollaborativeFiltering
from ..models.content_based_filtering import ContentBasedFiltering
from ..models.hybrid_recommender import HybridRecommender


class RecommendationService(IRecommendationService):
    """レコメンドサービス（Single Responsibility Principle）"""
    
    def __init__(self, data_provider: AnimeDataProvider):
        self.data_provider = data_provider
        self.strategies = {
            'collaborative': UserBasedCollaborativeFiltering(data_provider),
            'content': ContentBasedFiltering(data_provider),
            'hybrid': HybridRecommender(data_provider)
        }
        self.user_ratings_cache = {}
    
    def get_recommendations(self, user_id: int, strategy: str = "hybrid", n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """
        レコメンドを取得
        
        Args:
            user_id: ユーザーID
            strategy: レコメンド戦略 ('collaborative', 'content', 'hybrid')
            n_recommendations: レコメンド数
            
        Returns:
            レコメンドリスト
        """
        if strategy not in self.strategies:
            raise ValueError(f"サポートされていない戦略: {strategy}")
        
        try:
            recommendations = self.strategies[strategy].get_recommendations(user_id, n_recommendations)
            return recommendations
        except Exception as e:
            print(f"レコメンド生成エラー: {e}")
            return self._get_fallback_recommendations(n_recommendations)
    
    def add_rating(self, user_id: int, anime_id: int, rating: float) -> bool:
        """
        評価を追加（キャッシュに保存）
        
        Args:
            user_id: ユーザーID
            anime_id: アニメID
            rating: 評価値
            
        Returns:
            成功したかどうか
        """
        try:
            if user_id not in self.user_ratings_cache:
                self.user_ratings_cache[user_id] = []
            
            # 既存の評価を更新または新規追加
            existing_rating = None
            for i, existing in enumerate(self.user_ratings_cache[user_id]):
                if existing['anime_id'] == anime_id:
                    existing_rating = i
                    break
            
            new_rating = {
                'user_id': user_id,
                'anime_id': anime_id,
                'rating': rating
            }
            
            if existing_rating is not None:
                self.user_ratings_cache[user_id][existing_rating] = new_rating
            else:
                self.user_ratings_cache[user_id].append(new_rating)
            
            return True
        except Exception as e:
            print(f"評価追加エラー: {e}")
            return False
    
    def get_user_ratings(self, user_id: int) -> List[Dict[str, Any]]:
        """ユーザーの評価履歴を取得"""
        # キャッシュから取得
        if user_id in self.user_ratings_cache:
            return self.user_ratings_cache[user_id]
        
        # データプロバイダーから取得
        ratings_df = self.data_provider.get_user_ratings(user_id)
        if not ratings_df.empty:
            ratings = ratings_df.to_dict('records')
            self.user_ratings_cache[user_id] = ratings
            return ratings
        
        return []
    
    def get_anime_info(self, anime_id: int) -> Optional[Dict[str, Any]]:
        """アニメ情報を取得"""
        return self.data_provider.get_anime_info(anime_id)
    
    def get_popular_anime(self, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """人気アニメを取得"""
        popular_anime = self.data_provider.get_popular_anime(min_ratings=100)
        
        recommendations = []
        for _, anime in popular_anime.head(n_recommendations).iterrows():
            recommendations.append({
                'anime_id': anime['anime_id'],
                'name': anime.get('name', 'Unknown'),
                'genre': anime.get('genre', ''),
                'type': anime.get('type', ''),
                'episodes': anime.get('episodes', 0),
                'rating': anime.get('rating', 0),
                'recommendation_type': 'popular'
            })
        
        return recommendations
    
    def get_diverse_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """多様性を考慮したレコメンドを取得"""
        if 'hybrid' in self.strategies and hasattr(self.strategies['hybrid'], 'get_diverse_recommendations'):
            return self.strategies['hybrid'].get_diverse_recommendations(user_id, n_recommendations)
        else:
            return self.get_recommendations(user_id, 'hybrid', n_recommendations)
    
    def get_personalized_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """パーソナライズされたレコメンドを取得"""
        if 'hybrid' in self.strategies and hasattr(self.strategies['hybrid'], 'get_personalized_recommendations'):
            return self.strategies['hybrid'].get_personalized_recommendations(user_id, n_recommendations)
        else:
            return self.get_recommendations(user_id, 'hybrid', n_recommendations)
    
    def _get_fallback_recommendations(self, n_recommendations: int) -> List[Dict[str, Any]]:
        """フォールバック用の人気アニメ推薦"""
        return self.get_popular_anime(n_recommendations)
    
    def get_available_strategies(self) -> List[str]:
        """利用可能なレコメンド戦略を取得"""
        return list(self.strategies.keys())
    
    def get_strategy_info(self, strategy: str) -> Dict[str, Any]:
        """レコメンド戦略の情報を取得"""
        if strategy not in self.strategies:
            return {}
        
        strategy_info = {
            'name': strategy,
            'description': self._get_strategy_description(strategy),
            'available': True
        }
        
        return strategy_info
    
    def _get_strategy_description(self, strategy: str) -> str:
        """戦略の説明を取得"""
        descriptions = {
            'collaborative': 'ユーザーベース協調フィルタリング - 類似ユーザーの評価パターンに基づく推薦',
            'content': 'コンテンツベースフィルタリング - アニメの特徴（ジャンル、タイプ等）に基づく推薦',
            'hybrid': 'ハイブリッド推薦 - 協調フィルタリングとコンテンツベースフィルタリングを組み合わせた推薦'
        }
        return descriptions.get(strategy, '説明なし') 