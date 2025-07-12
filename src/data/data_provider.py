"""
データプロバイダーの実装（Kaggleデータ対応）
Interface Segregation Principleを実装
"""
import pandas as pd
import numpy as np
import os
from typing import Any, Optional
from ..interfaces.recommendation_interface import IDataProvider


class AnimeDataProvider(IDataProvider):
    """アニメデータプロバイダー（Single Responsibility Principle）"""
    
    def __init__(self, data_dir: str = "data", sample_size: Optional[int] = None):
        self.data_dir = data_dir
        self.sample_size = sample_size  # サンプリングサイズ（Noneの場合は全データ）
        self._anime_data: Optional[pd.DataFrame] = None
        self._rating_data: Optional[pd.DataFrame] = None
        self._user_data: Optional[pd.DataFrame] = None
    
    def load_anime_data(self) -> pd.DataFrame:
        """アニメデータを読み込み（anime_with_synopsis.csvを利用）"""
        if self._anime_data is None:
            file_path = os.path.join(self.data_dir, "anime_with_synopsis.csv")
            if os.path.exists(file_path):
                self._anime_data = pd.read_csv(file_path)
                # カラム名を統一
                self._anime_data = self._anime_data.rename(columns={
                    'MAL_ID': 'anime_id',
                    'Name': 'name',
                    'Score': 'rating',
                    'Genres': 'genre',
                    'sypnopsis': 'synopsis'
                })
                print(f"アニメデータを読み込みました: {len(self._anime_data)}件")
            else:
                raise FileNotFoundError(f"アニメデータファイルが見つかりません: {file_path}")
        return self._anime_data
    
    def load_rating_data(self) -> pd.DataFrame:
        """評価データを読み込み（rating_complete.csvを利用、サンプリング対応）"""
        if self._rating_data is None:
            file_path = os.path.join(self.data_dir, "rating_complete.csv")
            if os.path.exists(file_path):
                if self.sample_size:
                    # サンプリングサイズが指定されている場合はサンプリング
                    print(f"評価データをサンプリング中... (サンプルサイズ: {self.sample_size})")
                    # ランダムサンプリング
                    self._rating_data = pd.read_csv(file_path, nrows=self.sample_size)
                else:
                    # 全データを読み込み
                    self._rating_data = pd.read_csv(file_path)
                
                # カラム名を統一
                self._rating_data = self._rating_data.rename(columns={
                    'user_id': 'user_id',
                    'anime_id': 'anime_id',
                    'rating': 'rating'
                })
                print(f"評価データを読み込みました: {len(self._rating_data)}件")
            else:
                raise FileNotFoundError(f"評価データファイルが見つかりません: {file_path}")
        return self._rating_data
    
    def load_user_data(self) -> pd.DataFrame:
        """ユーザーデータを読み込み（animelist.csvから生成）"""
        if self._user_data is None:
            file_path = os.path.join(self.data_dir, "animelist.csv")
            if os.path.exists(file_path):
                if self.sample_size:
                    # サンプリングサイズが指定されている場合はサンプリング
                    df = pd.read_csv(file_path, usecols=['user_id'], nrows=self.sample_size)
                else:
                    df = pd.read_csv(file_path, usecols=['user_id'])
                self._user_data = df[['user_id']].drop_duplicates().reset_index(drop=True)
                print(f"ユーザーデータをanimelist.csvから生成しました: {len(self._user_data)}件")
            else:
                # 評価データからユーザー情報を生成
                rating_data = self.load_rating_data()
                self._user_data = rating_data[['user_id']].drop_duplicates().reset_index(drop=True)
                print(f"評価データからユーザー情報を生成しました: {len(self._user_data)}件")
        return self._user_data
    
    def get_anime_info(self, anime_id: int) -> Optional[dict]:
        """アニメ情報を取得（シノプシス・ジャンル含む）"""
        anime_data = self.load_anime_data()
        anime_info = anime_data[anime_data['anime_id'] == anime_id]
        if not anime_info.empty:
            return anime_info.iloc[0].to_dict()
        return None
    
    def get_user_ratings(self, user_id: int) -> pd.DataFrame:
        """ユーザーの評価履歴を取得"""
        rating_data = self.load_rating_data()
        return rating_data[rating_data['user_id'] == user_id]
    
    def get_anime_ratings(self, anime_id: int) -> pd.DataFrame:
        """アニメの評価履歴を取得"""
        rating_data = self.load_rating_data()
        return rating_data[rating_data['anime_id'] == anime_id]
    
    def get_popular_anime(self, min_ratings: int = 100) -> pd.DataFrame:
        """人気アニメを取得"""
        rating_data = self.load_rating_data()
        anime_data = self.load_anime_data()
        
        # 評価数の多いアニメを取得
        rating_counts = rating_data.groupby('anime_id').size().reset_index(name='rating_count')
        popular_anime_ids = rating_counts[rating_counts['rating_count'] >= min_ratings]['anime_id']
        
        return anime_data[anime_data['anime_id'].isin(popular_anime_ids)]
    
    def get_sample_users(self, n_users: int = 1000) -> list:
        """サンプルユーザーIDを取得"""
        user_data = self.load_user_data()
        if len(user_data) > n_users:
            return user_data['user_id'].sample(n_users).tolist()
        return user_data['user_id'].tolist()
    
    def get_sample_ratings(self, n_ratings: int = 100000) -> pd.DataFrame:
        """サンプル評価データを取得"""
        rating_data = self.load_rating_data()
        if len(rating_data) > n_ratings:
            return rating_data.sample(n_ratings)
        return rating_data 