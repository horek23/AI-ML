import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("../laba1/clean_dataset.csv")

# Разделение датасета на обучающую и тестовую выборки
X = df.drop(['SalePrice'], axis=1)
y = df['SalePrice']

# Классификация(дорогой или дешевый дом)
median_price = y.median()
y_class = (y > median_price).astype(int)
X_train_cl, X_test_cl, y_train_cl, y_test_cl = train_test_split(X, y_class, test_size=0.3, random_state=23)
dt_model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=23)
dt_model.fit(X_train_cl, y_train_cl)

# Вероятности классов (от 0 до 1)
y_probs = dt_model.predict_proba(X_test_cl)[:, 1]
# Метки классов (0 или 1)
y_pred = dt_model.predict(X_test_cl)

# Построения ROC-кривой
fpr, tpr, thresholds = roc_curve(y_test_cl, y_probs)
plt.plot(fpr, tpr, marker='o')
plt.ylim([0,1.1])
plt.xlim([0,1.1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC curve')
plt.show()

print(f"Площадь под ROC-кривой: {auc(fpr, tpr)}")

# Регрессия
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=23)
dt_regressor = DecisionTreeRegressor(max_depth=5, random_state=23)
dt_regressor.fit(X_train, y_train)
y_pred = dt_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
print(f"Коэффициент детерминации (R^2): {r2:.4f}")

# Граф дерева
tree.plot_tree(dt_regressor)
plt.show()