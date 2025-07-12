"""
Kaggleからアニメレコメンドデータセットをダウンロード
"""
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi


class DatasetDownloader:
    """データセットダウンローダー（Single Responsibility Principle）"""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        self.api = KaggleApi()
        self.api.authenticate()
    
    def download_anime_dataset(self):
        """アニメレコメンドデータセットをダウンロード"""
        try:
            # データセットのダウンロード
            self.api.dataset_download_files(
                'hernan4444/anime-recommendation-database-2020',
                path=self.output_dir,
                unzip=True
            )
            print(f"データセットが {self.output_dir} にダウンロードされました")
            
            # ファイル構造の確認
            self._list_downloaded_files()
            
        except Exception as e:
            print(f"ダウンロードエラー: {e}")
            print("Kaggle APIの設定を確認してください")
    
    def _list_downloaded_files(self):
        """ダウンロードされたファイルを一覧表示"""
        if os.path.exists(self.output_dir):
            print("\nダウンロードされたファイル:")
            for root, dirs, files in os.walk(self.output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    print(f"  {file_path} ({file_size} bytes)")


if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.download_anime_dataset() 