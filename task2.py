import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_blobs

# Создание случайных данных
X, y = make_blobs(n_samples=200, centers=4, random_state=0)

# Выбор признаков для визуализации
feature_indices = input("Введите индексы признаков через запятую (например, 0,1,2): ").split(",")
feature_indices = [int(idx) for idx in feature_indices]

# K-Means
kmeans = KMeans(n_clusters=4)
kmeans_labels = kmeans.fit_predict(X)

# Agglomerative clustering
agg_clustering = AgglomerativeClustering(n_clusters=4)
agg_labels = agg_clustering.fit_predict(X)

# DBSCAN
# eps - радиус точек
# min_sample - минимальное количество точек
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Визуализация результатов в сокращенном пространстве признаков
X_reduced = X[:, feature_indices]

fig = px.scatter(x=X_reduced[:, 0], y=X_reduced[:, 1], color=kmeans_labels)
fig.update_layout(title="K-Means Clustering")
fig.show()

fig = px.scatter(x=X_reduced[:, 0], y=X_reduced[:, 1], color=agg_labels)
fig.update_layout(title="Agglomerative Clustering")
fig.show()

fig = px.scatter(x=X_reduced[:, 0], y=X_reduced[:, 1], color=dbscan_labels)
fig.update_layout(title="DBSCAN Clustering")
fig.show()