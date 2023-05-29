import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import davies_bouldin_score
import plotly.express as px
import matplotlib.pyplot as plt


df = pd.read_csv('./datasets/transfusion.data.csv')
# print(df)

# first_feature = input('Select first features with the highest cross-correlation value:')
X = 'Recency'
# second_feature = input('Select second features with the highest cross-correlation value:')
Y = 'Frequency'

MAX_NUMBER_OF_CLASTERS = int(input())

davies_bouldin_score_data = [[], []]

for i in range(2, MAX_NUMBER_OF_CLASTERS+1):
    kmeans = KMeans(n_clusters=i, n_init=MAX_NUMBER_OF_CLASTERS+1)
    model = kmeans.fit_predict(df[[X, Y]])
    davies_bouldin_score_data[0].append(i)
    davies_bouldin_score_data[1].append(davies_bouldin_score(df[[X, Y]], model))

davies_bouldin_figure = plt.plot(davies_bouldin_score_data[0], davies_bouldin_score_data[1])
plt.xlabel('Количество класетров')
plt.ylabel('Davies-Bouldin Score')
plt.title('Davies-Bouldin Score for Different Number of Clusters')
plt.show()

# USER_NUMBER_OF_CLASTERS_CHOISE = int(input())
#
# kmeans = KMeans(n_clusters=USER_NUMBER_OF_CLASTERS_CHOISE)
# kmeans.fit_predict(df[[X, Y]])
# figure = plt.scatter(df[X], df[Y], c=kmeans.labels_, cmap='rainbow')
# figure
#
# davies_bouldin_score_data = [[], []]
#
# for i in range(2, 9 + 1):
#     agglomerative = AgglomerativeClustering(n_clusters=i)
#     model = agglomerative.fit_predict(df[[X, Y]])
#     davies_bouldin_score_data[0].append(i)
#     davies_bouldin_score_data[1].append(davies_bouldin_score(df[[X, Y]], model))
#
# davies_bouldin_figure = plt.plot(davies_bouldin_score_data[0], davies_bouldin_score_data[1])
# davies_bouldin_figure
#
#
# USER_NUMBER_OF_CLASTERS_CHOISE = int(input())
#
# agglomerative = AgglomerativeClustering(n_clusters=USER_NUMBER_OF_CLASTERS_CHOISE)
# agglomerative.fit_predict(df[[X, Y]])
# figure = plt.scatter(df[X], df[Y], c=agglomerative.labels_, cmap='rainbow')
# figure
#
#
# USER_NUMBER_OF_EPS_CHOISE = int(input())
#
# dbscan = DBSCAN(eps=USER_NUMBER_OF_EPS_CHOISE)
# dbscan.fit_predict(df[[X, Y]])
# figure = plt.scatter(df[X], df[Y], c=dbscan.labels_, cmap='rainbow')
# figure


