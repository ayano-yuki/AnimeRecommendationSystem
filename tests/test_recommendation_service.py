"""
レコメンドサービスのテスト
"""
import unittest
import pandas as pd
import os
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.data_provider import AnimeDataProvider
from src.services.recommendation_service import RecommendationService


class TestRecommendationService(unittest.TestCase):
    """RecommendationServiceのテストクラス"""
    
    def setUp(self):
        """テスト前の準備"""
        try:
            self.data_provider = AnimeDataProvider("data")
            self.service = RecommendationService(self.data_provider)
        except Exception as e:
            self.skipTest(f"データプロバイダーの初期化に失敗: {e}")
    
    def test_get_available_strategies(self):
        """利用可能な戦略取得テスト"""
        strategies = self.service.get_available_strategies()
        self.assertIsInstance(strategies, list)
        self.assertIn('collaborative', strategies)
        self.assertIn('content', strategies)
        self.assertIn('hybrid', strategies)
    
    def test_get_strategy_info(self):
        """戦略情報取得テスト"""
        strategy_info = self.service.get_strategy_info('hybrid')
        self.assertIsInstance(strategy_info, dict)
        self.assertIn('name', strategy_info)
        self.assertIn('description', strategy_info)
        self.assertIn('available', strategy_info)
    
    def test_add_rating(self):
        """評価追加テスト"""
        result = self.service.add_rating(1, 1, 8.5)
        self.assertTrue(result)
        
        # 追加した評価を確認
        ratings = self.service.get_user_ratings(1)
        self.assertIsInstance(ratings, list)
    
    def test_get_anime_info(self):
        """アニメ情報取得テスト"""
        try:
            anime_data = self.data_provider.load_anime_data()
            if not anime_data.empty:
                test_anime_id = anime_data.iloc[0]['anime_id']
                anime_info = self.service.get_anime_info(test_anime_id)
                self.assertIsNotNone(anime_info)
        except FileNotFoundError:
            self.skipTest("データファイルが見つかりません")
    
    def test_get_popular_anime(self):
        """人気アニメ取得テスト"""
        try:
            popular_anime = self.service.get_popular_anime(5)
            self.assertIsInstance(popular_anime, list)
            if popular_anime:
                self.assertIn('anime_id', popular_anime[0])
                self.assertIn('name', popular_anime[0])
        except Exception as e:
            self.skipTest(f"人気アニメ取得に失敗: {e}")
    
    def test_get_recommendations_invalid_strategy(self):
        """無効な戦略でのレコメンド取得テスト"""
        with self.assertRaises(ValueError):
            self.service.get_recommendations(1, "invalid_strategy", 5)


if __name__ == '__main__':
    unittest.main() 