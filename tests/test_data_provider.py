"""
データプロバイダーのテスト
"""
import unittest
import pandas as pd
import os
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.data_provider import AnimeDataProvider


class TestAnimeDataProvider(unittest.TestCase):
    """AnimeDataProviderのテストクラス"""
    
    def setUp(self):
        """テスト前の準備"""
        self.data_provider = AnimeDataProvider("data")
    
    def test_load_anime_data(self):
        """アニメデータ読み込みテスト"""
        try:
            anime_data = self.data_provider.load_anime_data()
            self.assertIsInstance(anime_data, pd.DataFrame)
            if not anime_data.empty:
                self.assertIn('anime_id', anime_data.columns)
                self.assertIn('name', anime_data.columns)
        except FileNotFoundError:
            self.skipTest("データファイルが見つかりません")
    
    def test_load_rating_data(self):
        """評価データ読み込みテスト"""
        try:
            rating_data = self.data_provider.load_rating_data()
            self.assertIsInstance(rating_data, pd.DataFrame)
            if not rating_data.empty:
                self.assertIn('user_id', rating_data.columns)
                self.assertIn('anime_id', rating_data.columns)
                self.assertIn('rating', rating_data.columns)
        except FileNotFoundError:
            self.skipTest("データファイルが見つかりません")
    
    def test_get_anime_info(self):
        """アニメ情報取得テスト"""
        try:
            anime_data = self.data_provider.load_anime_data()
            if not anime_data.empty:
                test_anime_id = anime_data.iloc[0]['anime_id']
                anime_info = self.data_provider.get_anime_info(test_anime_id)
                self.assertIsNotNone(anime_info)
                self.assertIn('anime_id', anime_info)
        except FileNotFoundError:
            self.skipTest("データファイルが見つかりません")
    
    def test_get_user_ratings(self):
        """ユーザー評価取得テスト"""
        try:
            rating_data = self.data_provider.load_rating_data()
            if not rating_data.empty:
                test_user_id = rating_data.iloc[0]['user_id']
                user_ratings = self.data_provider.get_user_ratings(test_user_id)
                self.assertIsInstance(user_ratings, pd.DataFrame)
        except FileNotFoundError:
            self.skipTest("データファイルが見つかりません")


if __name__ == '__main__':
    unittest.main() 