import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import davies_bouldin_score
import plotly.express as px
import matplotlib.pyplot as plt
from warnings import simplefilter
# игнорировать все предупреждения о будущих изменениях
simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('./datasets/transfusion.data.csv')
#df = pd.read_csv('./datasets/r5.csv')
#df = df.drop(['id'], axis=1)
#df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
# Построение матрицы корреляции
#df = df.drop(['id'], axis=1)
#df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
# Построение матрицы корреляции
X = 'Recency'
# second_feature = input('Select second features with the highest cross-correlation value:')
Y = 'Frequency'

USER_START_CHOICE = int(input())
USER_STOP_CHOICE = int(input())
USER_STEP_CHOICE = int(input())

davies_bouldin_score_data = [[], []]
j = 1
for i in range(USER_START_CHOICE, USER_STOP_CHOICE + 1, USER_STEP_CHOICE):
    dbscan = DBSCAN(eps=i)
    model = dbscan.fit_predict(df[[X, Y]])
    j =j+1
    davies_bouldin_score_data[0].append(j)
    davies_bouldin_score_data[1].append(davies_bouldin_score(df[[X, Y]], model))

davies_bouldin_figure = plt.plot(davies_bouldin_score_data[0], davies_bouldin_score_data[1])
plt.show()

USER_NUMBER_OF_EPS_CHOISE = int(input())

dbscan = DBSCAN(eps=USER_NUMBER_OF_EPS_CHOISE)
dbscan.fit_predict(df[[X, Y]])
figure = plt.scatter(df[X], df[Y], c=dbscan.labels_, cmap='rainbow')
plt.show()

MAX_NUMBER_OF_CLASTERS = int(input())

davies_bouldin_score_data = [[], []]
# Расчет индекса Дэвиса-Болдина для различного количества кластеров
for i in range(2, MAX_NUMBER_OF_CLASTERS+1):
    kmeans = KMeans(n_clusters=i)
    model = kmeans.fit_predict(df[[X, Y]])
    davies_bouldin_score_data[0].append(i)
    davies_bouldin_score_data[1].append(davies_bouldin_score(df[[X, Y]], model))
# Визуализация индекса Дэвиса-Болдина
davies_bouldin_figure = plt.plot(davies_bouldin_score_data[0], davies_bouldin_score_data[1])
plt.xlabel('Количество класетров')
plt.ylabel('Davies-Bouldin Score')
plt.title('Davies-Bouldin Score for Different Number of Clusters')
plt.show()

USER_NUMBER_OF_CLASTERS_CHOISE = int(input())
# Кластеризация методом KMeans
kmeans = KMeans(n_clusters=USER_NUMBER_OF_CLASTERS_CHOISE)
kmeans.fit_predict(df[[X, Y]])
figure = plt.scatter(df[X], df[Y], c=kmeans.labels_, cmap='rainbow')
plt.show()

davies_bouldin_score_data = [[], []]

# Расчет индекса Дэвиса-Болдина для разного количества кластеров (метод иерархической кластеризации)
for i in range(2, USER_NUMBER_OF_CLASTERS_CHOISE + 1):
    agglomerative = AgglomerativeClustering(n_clusters=i)
    model = agglomerative.fit_predict(df[[X, Y]])
    davies_bouldin_score_data[0].append(i)
    davies_bouldin_score_data[1].append(davies_bouldin_score(df[[X, Y]], model))

davies_bouldin_figure = plt.plot(davies_bouldin_score_data[0], davies_bouldin_score_data[1])
plt.show()


USER_NUMBER_OF_CLASTERS_CHOISE = int(input())
# Кластеризация методом иерархической кластеризации
agglomerative = AgglomerativeClustering(n_clusters=USER_NUMBER_OF_CLASTERS_CHOISE)
agglomerative.fit_predict(df[[X, Y]])
figure = plt.scatter(df[X], df[Y], c=agglomerative.labels_, cmap='rainbow')
plt.show()


davies_bouldin_score_data = [[], []]

for i in range(2, MAX_NUMBER_OF_CLASTERS + 1):
    agglomerative = AgglomerativeClustering(n_clusters=i)
    model = agglomerative.fit_predict(df[[X, Y]])
    davies_bouldin_score_data[0].append(i)
    davies_bouldin_score_data[1].append(davies_bouldin_score(df[[X, Y]], model))

davies_bouldin_figure = plt.plot(davies_bouldin_score_data[0], davies_bouldin_score_data[1])
plt.show()



