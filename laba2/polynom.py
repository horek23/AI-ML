import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("../laba1/clean_dataset.csv")
X = df.drop(['SalePrice'], axis=1)
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=23)
linear_model = LinearRegression()

# Полиномиальная регрессия
# Полином просто увеличивает количество признаков
# то есть если в линейной регрессии было 2 признака (a,b)
# то в полиномиальной 2 степени будет столько признаков (1,a,a^2,ab,b,b^2)

n = 2
poly_features = PolynomialFeatures(n, interaction_only=True, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)
linear_model.fit(X_train_poly, y_train)
y_pred_poly = linear_model.predict(X_test_poly)
RMSE = root_mean_squared_error(y_test, y_pred_poly)
MAE = mean_absolute_error(y_test, y_pred_poly)
print(f"Средняя абсолютная ошибка: {MAE}")
print(f"Корень среднеквадратичной ошибки: {RMSE}\n")

# Регуляризация
best_model_1 = ElasticNet(alpha=1, l1_ratio=0.85, max_iter=40000)
best_model_1.fit(X_train_poly, y_train)
y_pred_poly = best_model_1.predict(X_test_poly)
RMSE = root_mean_squared_error(y_test, y_pred_poly)
MAE = mean_absolute_error(y_test, y_pred_poly)
print(f"Средняя абсолютная ошибка: {MAE}")
print(f"Корень среднеквадратичной ошибки: {RMSE}\n")