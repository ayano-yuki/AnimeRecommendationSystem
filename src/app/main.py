"""
アニメレコメンドシステムのStreamlitアプリケーション
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.data_provider import AnimeDataProvider
from src.services.recommendation_service import RecommendationService


def initialize_session_state():
    """セッション状態を初期化"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = 1
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = []
    if 'sample_size' not in st.session_state:
        st.session_state.sample_size = 100000  # デフォルトサンプルサイズ


def load_data(sample_size: int = 100000):
    """データを読み込み"""
    try:
        data_provider = AnimeDataProvider(sample_size=sample_size)
        service = RecommendationService(data_provider)
        return data_provider, service
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return None, None


def display_header():
    """ヘッダーを表示"""
    st.set_page_config(
        page_title="アニメレコメンドシステム",
        page_icon="🎌",
        layout="wide"
    )
    
    st.title("🎌 アニメレコメンドシステム")
    st.markdown("Kaggleの「Anime Recommendation Database 2020」を使用したレコメンドシステム")
    
    # サイドバーにユーザーID設定とサンプリングサイズ設定
    with st.sidebar:
        st.header("👤 ユーザー設定")
        user_id = st.number_input("ユーザーID", min_value=1, value=st.session_state.user_id)
        st.session_state.user_id = user_id
        
        st.header("⚙️ パフォーマンス設定")
        sample_size = st.selectbox(
            "データサンプルサイズ",
            [50000, 100000, 200000, 500000, 1000000],
            index=1,
            help="メモリ使用量を調整するためのサンプリングサイズ"
        )
        st.session_state.sample_size = sample_size
        
        st.info(f"現在のサンプルサイズ: {sample_size:,}件")


def display_recommendations(service: RecommendationService, strategy: str = "hybrid"):
    """レコメンドを表示"""
    st.header("🎯 おすすめアニメ")
    
    # レコメンド戦略の選択
    col1, col2 = st.columns([1, 3])
    with col1:
        strategy = st.selectbox(
            "レコメンド手法",
            ["hybrid", "collaborative", "content"],
            format_func=lambda x: {
                "hybrid": "ハイブリッド",
                "collaborative": "協調フィルタリング",
                "content": "コンテンツベース"
            }[x]
        )
    
    with col2:
        n_recommendations = st.slider("推薦数", min_value=5, max_value=20, value=10)
    
    # レコメンドを取得
    if st.button("レコメンドを取得", type="primary"):
        with st.spinner("レコメンドを生成中..."):
            try:
                recommendations = service.get_recommendations(
                    st.session_state.user_id, 
                    strategy, 
                    n_recommendations
                )
                st.session_state.recommendations = recommendations
                st.success(f"{len(recommendations)}件のレコメンドを生成しました！")
            except Exception as e:
                st.error(f"レコメンド生成エラー: {e}")
                st.info("メモリ不足の可能性があります。サンプルサイズを小さくしてください。")
    
    # レコメンド結果を表示
    if st.session_state.recommendations:
        display_recommendation_cards(st.session_state.recommendations)


def display_recommendation_cards(recommendations: List[Dict[str, Any]]):
    """レコメンドカードを表示"""
    st.subheader("📺 推薦アニメ一覧")
    
    # カード形式で表示
    cols = st.columns(2)
    for i, rec in enumerate(recommendations):
        col_idx = i % 2
        with cols[col_idx]:
            with st.container():
                st.markdown(f"""
                <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <h4>{rec.get('name', 'Unknown')}</h4>
                    <p><strong>ジャンル:</strong> {rec.get('genre', 'N/A')}</p>
                    <p><strong>タイプ:</strong> {rec.get('type', 'N/A')}</p>
                    <p><strong>エピソード数:</strong> {rec.get('episodes', 'N/A')}</p>
                    <p><strong>評価:</strong> {rec.get('rating', 'N/A')}</p>
                    <p><strong>推薦タイプ:</strong> {rec.get('recommendation_type', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)


def display_user_ratings(service: RecommendationService):
    """ユーザー評価を表示"""
    st.header("⭐ あなたの評価履歴")
    
    # 評価履歴を取得
    user_ratings = service.get_user_ratings(st.session_state.user_id)
    
    if user_ratings:
        # 評価データをDataFrameに変換
        ratings_df = pd.DataFrame(user_ratings)
        
        # 評価分布の可視化
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                ratings_df, 
                x='rating', 
                title="評価分布",
                nbins=10
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # 評価の統計情報
            st.subheader("📊 評価統計")
            st.metric("評価数", len(ratings_df))
            st.metric("平均評価", f"{ratings_df['rating'].mean():.2f}")
            st.metric("最高評価", ratings_df['rating'].max())
            st.metric("最低評価", ratings_df['rating'].min())
        
        # 評価一覧
        st.subheader("📝 評価一覧")
        for rating in user_ratings:
            anime_info = service.get_anime_info(rating['anime_id'])
            if anime_info:
                st.write(f"**{anime_info.get('name', 'Unknown')}** - 評価: {rating['rating']}")
    else:
        st.info("まだ評価履歴がありません。")


def display_anime_search(service: RecommendationService):
    """アニメ検索機能"""
    st.header("🔍 アニメ検索")
    
    # 検索機能
    search_term = st.text_input("アニメ名を入力してください")
    
    if search_term:
        # 簡易的な検索機能（実際の実装ではより高度な検索が必要）
        try:
            anime_data = service.data_provider.load_anime_data()
            search_results = anime_data[
                anime_data['name'].str.contains(search_term, case=False, na=False)
            ].head(10)
            
            if not search_results.empty:
                st.subheader("検索結果")
                for _, anime in search_results.iterrows():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{anime['name']}**")
                        st.write(f"ジャンル: {anime.get('genre', 'N/A')}")
                    with col2:
                        st.write(f"評価: {anime.get('rating', 'N/A')}")
                    with col3:
                        if st.button(f"評価", key=f"rate_{anime['anime_id']}"):
                            # 評価機能（簡易版）
                            st.info("評価機能は開発中です")
            else:
                st.warning("該当するアニメが見つかりませんでした。")
        except Exception as e:
            st.error(f"検索エラー: {e}")


def display_popular_anime(service: RecommendationService):
    """人気アニメを表示"""
    st.header("🔥 人気アニメ")
    
    if st.button("人気アニメを表示"):
        with st.spinner("人気アニメを取得中..."):
            try:
                popular_anime = service.get_popular_anime(10)
                
                # 人気アニメの可視化
                if popular_anime:
                    df = pd.DataFrame(popular_anime)
                    
                    fig = px.bar(
                        df, 
                        x='name', 
                        y='rating',
                        title="人気アニメ（評価順）",
                        labels={'name': 'アニメ名', 'rating': '評価'}
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 詳細一覧
                    st.subheader("📋 人気アニメ一覧")
                    for anime in popular_anime:
                        st.write(f"**{anime['name']}** - 評価: {anime['rating']} - ジャンル: {anime['genre']}")
                
            except Exception as e:
                st.error(f"人気アニメ取得エラー: {e}")


def main():
    """メイン関数"""
    initialize_session_state()
    display_header()
    
    # データ読み込み（サンプリングサイズを指定）
    data_provider, service = load_data(st.session_state.sample_size)
    
    if data_provider is None or service is None:
        st.error("データの読み込みに失敗しました。データセットが正しくダウンロードされているか確認してください。")
        return
    
    # タブ形式で機能を表示
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 レコメンド", "⭐ 評価履歴", "🔍 アニメ検索", "🔥 人気アニメ"])
    
    with tab1:
        display_recommendations(service)
    
    with tab2:
        display_user_ratings(service)
    
    with tab3:
        display_anime_search(service)
    
    with tab4:
        display_popular_anime(service)
    
    # フッター
    st.markdown("---")
    st.markdown("**アニメレコメンドシステム** - Kaggleデータセットを使用")


if __name__ == "__main__":
    main() 