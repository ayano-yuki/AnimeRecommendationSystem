"""
レコメンドシステムの基底インターフェース
Dependency Inversion Principleを実装
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class IRecommendationStrategy(ABC):
    """レコメンド戦略の基底インターフェース"""
    
    @abstractmethod
    def get_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """
        ユーザーに対するレコメンドを取得
        
        Args:
            user_id: ユーザーID
            n_recommendations: レコメンド数
            
        Returns:
            レコメンドリスト
        """
        pass


class IDataProvider(ABC):
    """データプロバイダーの基底インターフェース"""
    
    @abstractmethod
    def load_anime_data(self) -> Any:
        """アニメデータを読み込み"""
        pass
    
    @abstractmethod
    def load_rating_data(self) -> Any:
        """評価データを読み込み"""
        pass
    
    @abstractmethod
    def load_user_data(self) -> Any:
        """ユーザーデータを読み込み"""
        pass


class IModelEvaluator(ABC):
    """モデル評価の基底インターフェース"""
    
    @abstractmethod
    def evaluate_model(self, model: IRecommendationStrategy, test_data: Any) -> Dict[str, float]:
        """
        モデルの評価を実行
        
        Args:
            model: 評価対象のモデル
            test_data: テストデータ
            
        Returns:
            評価指標の辞書
        """
        pass


class IRecommendationService(ABC):
    """レコメンドサービスの基底インターフェース"""
    
    @abstractmethod
    def get_recommendations(self, user_id: int, strategy: str = "hybrid", n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """
        レコメンドを取得
        
        Args:
            user_id: ユーザーID
            strategy: レコメンド戦略
            n_recommendations: レコメンド数
            
        Returns:
            レコメンドリスト
        """
        pass
    
    @abstractmethod
    def add_rating(self, user_id: int, anime_id: int, rating: float) -> bool:
        """
        評価を追加
        
        Args:
            user_id: ユーザーID
            anime_id: アニメID
            rating: 評価値
            
        Returns:
            成功したかどうか
        """
        pass 