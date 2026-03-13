import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("../laba1/clean_dataset.csv")

# Разделение датасета на обучающую и тестовую выборки
X = df.drop(['SalePrice'], axis=1)
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=23)

# Регрессия
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_test = linear_model.predict(X_test)

# Оценка регрессии
# RMSE сильно штрафует за большие ошибки
# RMSE>MAE - модель делает большие промахи, RMSE=MAE - модель стабильна и делает одинаковые ошибки

RMSE = root_mean_squared_error(y_test, y_pred_test)
MAE = mean_absolute_error(y_test, y_pred_test)
print(f"Средняя абсолютная ошибка: {MAE}")
print(f"Корень среднеквадратичной ошибки: {RMSE}\n")

# Улучшение обобщающей способности модели (регуляризация)
# L1 → зануляет лишние признаки
# L2 → уменьшает веса
# ElasticNet (L1+L2) → делает оба
# λ — это сила штрафа (ее подбирают, сравнивая ошибку)
# l1_ratio - задает соотношение L1 и L2 (1 - только L1; 0 - только L2; 0,5 - поровну и так далее)
# l1_ratio я просто перебирала вручную, пока не станет минимальная ошибка
# max_iter — это ограничение на количество шагов обучения, которые алгоритм может сделать, пока ищет лучшие веса модели
# число итераций ограничивают, чтобы алгоритм не работал бесконечно

alphas = [0.001, 0.01, 0.1, 1, 10]
best_alpha = None
best_rmse = float("inf")
for a in alphas:
    model = ElasticNet(alpha=a, l1_ratio=0.85, max_iter=40000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print("alpha:", a, "RMSE:", rmse)
    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = a
print("Лучшее alpha:", best_alpha)
print("Лучший RMSE:", best_rmse, "\n")

best_model = ElasticNet(alpha=0.1, l1_ratio=0.85, max_iter=40000)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
RMSE = root_mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
print(f"Средняя абсолютная ошибка: {MAE}")
print(f"Корень среднеквадратичной ошибки: {RMSE}\n")

# Приминение полиномиальной регрессии не дало улучшения модели,
# очень сильно возросло время обучения, а ошибка не уменьшилась.
# И даже когда я, убрала произведения признаков это не помогло уменьшить время обучения.
# Поэтому наилучшим вариантом для моего датасета является линейная регрессия с регуляризацией

# Классификация
median_price = y.median()
y_class = (y > median_price).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.3, random_state=23)
logreg_model = LogisticRegression(max_iter=3000)
logreg_model.fit(X_train, y_train)
y_pred_test = logreg_model.predict(X_test)

# Оценка классификации
report = classification_report(y_test, y_pred_test)
print(report)
