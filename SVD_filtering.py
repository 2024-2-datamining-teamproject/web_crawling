import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

# MovieLens 데이터 로드
ratings = pd.read_csv('ratings_copy.csv')  # 'ratings.csv' 파일에는 'userId', 'movieId', 'rating' 컬럼이 있어야 합니다.

# 사용자-아이템 평점 행렬 생성
ratings_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
original_ratings_matrix = ratings_matrix.copy()  # 원본 평점 행렬을 저장하여 평가 여부 확인용
ratings_matrix = ratings_matrix.fillna(0)  # SVD용으로 빈 값 채우기

# TruncatedSVD를 사용한 행렬 분해
svd = TruncatedSVD(n_components=100, random_state=42)
# user_factors = svd.fit_transform(ratings_matrix)  # 사용자 잠재 요인 행렬
# item_factors = svd.components_.T  # 아이템 잠재 요인 행렬

# # 사용자-아이템 예상 평점 계산
# predicted_ratings = np.dot(user_factors, item_factors.T)

# predicted_ratings = np.clip(predicted_ratings, 0, 5)


# 중심화된 평점 행렬로 SVD 적용
ratings_matrix_centered = ratings_matrix.sub(ratings_matrix.mean(axis=1), axis=0)
user_factors = svd.fit_transform(ratings_matrix_centered)
item_factors = svd.components_.T
predicted_ratings = np.dot(user_factors, item_factors.T) + ratings_matrix.mean(axis=1).values[:, np.newaxis]
predicted_ratings = np.clip(predicted_ratings, 0, 5)

def recommend_by_prediction(user_id, top_n=10):
    """
    SVD로 예측된 평점 기반으로 특정 사용자가 평가하지 않은 영화 중 예상 평점이 높은 상위 N개 추천
    """
    user_idx = user_id - 1  # 인덱스는 0부터 시작하므로 user_id - 1
    user_ratings = predicted_ratings[user_idx]
    
    # 사용자가 실제로 평가한 영화 제외
    rated_movies = original_ratings_matrix.loc[user_id][original_ratings_matrix.loc[user_id].notna()].index
    unrated_movies = [movie for movie in ratings_matrix.columns if movie not in rated_movies]
    
    # 예상 평점 계산 및 정렬
    recommendations = {movie: user_ratings[ratings_matrix.columns.get_loc(movie)] for movie in unrated_movies}
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    
    # 상위 N개의 추천 영화 반환
    return sorted_recommendations[:top_n]


# K-최근접 이웃 모델 설정 (코사인 유사도 사용, 상위 5명의 유사한 사용자 찾기)
knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5, n_jobs=-1)
knn_model.fit(ratings_matrix)
def recommend_by_knn(user_id, top_n=10):
    """
    KNN을 통해 유사한 사용자가 높게 평가한 것으로 예측된 영화를 추천
    """
    user_idx = ratings_matrix.index.get_loc(user_id)  # user_id에 해당하는 인덱스 찾기
    
    # 현재 사용자의 유사한 상위 K명의 사용자 인덱스 찾기
    _, indices = knn_model.kneighbors([predicted_ratings[user_idx]], n_neighbors=5)
    similar_users = ratings_matrix.index[indices.flatten()].tolist()

    # 유사한 사용자가 예측 평점이 높은 영화 추천
    recommendations = {}
    #print(similar_users)
    for similar_user in similar_users:
        similar_user_ratings = predicted_ratings[ratings_matrix.index.get_loc(similar_user)]
        
        # 현재 사용자가 실제로 평가하지 않은 영화만 추천 후보로 추가
        for movie_id, rating in enumerate(similar_user_ratings):
            if pd.isna(original_ratings_matrix.loc[user_id, ratings_matrix.columns[movie_id]]):  # 원본 행렬에서 실제 평점이 비어있는 경우
                if ratings_matrix.columns[movie_id] not in recommendations:
                    recommendations[ratings_matrix.columns[movie_id]] = rating
                else:
                    recommendations[ratings_matrix.columns[movie_id]] += rating  # 여러 유사 사용자가 평가한 경우 평점 누적

    # 예측 평점이 높은 순으로 정렬하여 상위 N개 추천
    # recommended_movies = [movie_id for movie_id, _ in sorted(recommendations.items(), key=lambda x: x[1], reverse=True)]
    recommended_movies = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return recommended_movies[:top_n]



# print("Recommendations for User 1 by Prediction:", recommend_by_prediction(user_id=1))
# print("Recommendations for User 1 by KNN:", recommend_by_knn(user_id=2))

# 예측된 평점 확인
user_id = 1
user_predicted_ratings = predicted_ratings[user_id - 1]

for rating in user_predicted_ratings[:20]:
    print(rating)