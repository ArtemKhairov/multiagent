
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_csv('./datasets/storms.csv')
# Remove string columns
df = df.drop(['name', 'status', 'category', 'tropicalstorm_force_diameter', 'hurricane_force_diameter'], axis=1)

corr_matrix = df.corr(method='pearson').round(3)
# Вывод матрицы корреляции в консоль
print(corr_matrix)

# Запись матрицы корреляции в файл
corr_matrix.to_csv('corr_matrix.csv', sep=',', index=False)

# визуализация тепловой карты
plt.matshow(corr_matrix, cmap='coolwarm')
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.colorbar()
plt.title('Матрица корреляций')
plt.show()

# преобразование матрицы корреляций в Series
corr_series = corr_matrix.unstack().sort_values(ascending=False)

# отбор значений с корреляцией между разными признаками
corr_pairs = corr_series[corr_series != 1]

# выбор двух признаков с наибольшей взаимной корреляцией
sorted_pairs = corr_pairs.abs().sort_values(ascending=False)
top_corr_pairs = sorted_pairs.head(2)
#top_corr_pairs = corr_pairs.head(2)
#sorted_pairs = sorted(top_corr_pairs.items(), key=lambda x: abs(x[1]), reverse=True)

# вывод на экран и предложение выбрать признаки
print("Выберите два признака с наибольшей взаимной корреляцией:")
for i, (pair, corr) in enumerate(top_corr_pairs.items()):
#for i, (pair, corr) in enumerate(sorted_pairs):
    feature1, feature2 = pair
    print(f"{i+1}. {feature1} и {feature2} (коэффициент корреляции: {abs(corr):.2f})")

# построение графика
plt.scatter(df[feature1], df[feature2])
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title('Данные выборки')
plt.show()

# Обучаем модель
X_train_size = int(len(df[feature1]) * 0.2)
Y_train_size = int(len(df[feature2]) * 0.2)

X_train = df[feature1].iloc[:X_train_size].values.reshape(X_train_size, 1)
X_test = df[feature1].iloc[X_train_size:].values.reshape(len(df[feature1]) - X_train_size, 1)

Y_train = df[feature2].iloc[:Y_train_size].values.reshape(Y_train_size, 1)
Y_test = df[feature2].iloc[Y_train_size:].values.reshape(len(df[feature2]) - Y_train_size, 1)

model = LinearRegression()
model.fit(X_train, Y_train)
score = model.score(X_train, Y_train)

# Рассчитываем коэффициенты наклона и точку пересечения
coef = model.coef_
intercept = model.intercept_
print(coef)
print(intercept)
#Получение предсказаний на тестовой выборке
Y_predict = model.predict(X_test)

plt.scatter(X_test, Y_test, color="black")
plt.plot(X_test, Y_predict, color="blue", linewidth=3)
plt.show()

# сравниваем предсказываемые значения с с фактическими из тестовой
# средняя квадратичная ошибка
mse = mean_squared_error(Y_test, Y_predict)
#Средняя абсолютная ошибка
mae = mean_absolute_error(Y_test, Y_predict)

print("Оценка модели:")
print("Коэффициент детерминации (R^2):", score)
print("Среднеквадратичная ошибка (MSE):", mse)
print("Средняя абсолютная ошибка (MAE):", mae)

# Вводимое значение x
X_input = int(input())
Y_input_predict = model.predict(np.array([X_input]).reshape(-1, 1))
print(Y_input_predict)