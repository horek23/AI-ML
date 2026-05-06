import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

df = pd.read_csv("../laba1/clean_dataset.csv")

X = df.drop(['SalePrice'], axis=1)
y = df['SalePrice']
median_price = y.median()
y_class = (y > median_price).astype(int)

# Случайный лес (n_estimators=100 — количество деревьев в лесу, oob_score=True — ОБЯЗАТЕЛЬНО для оценки через OOB)
model = RandomForestClassifier(n_estimators=100, max_depth=5, oob_score=True, random_state=23)
model.fit(X, y_class)

# Оценка модели через OOB данные
print("Доля правильных ответов:", model.oob_score_)
print("Out-Of-Bag Error : ", 1 - model.oob_score_)

# Решение задачи классификации методом AdaBoost
X_train_cl, X_test_cl, y_train_cl, y_test_cl = train_test_split(X, y_class, test_size=0.3, random_state=23)
ada_model = AdaBoostClassifier(n_estimators=100, random_state=23)
ada_model.fit(X_train_cl, y_train_cl)

y_pred = ada_model.predict(X_test_cl)
y_probs = ada_model.predict_proba(X_test_cl)[:, 1]

# Построения ROC-кривой для классификации методом AdaBoost
fpr, tpr, thresholds = roc_curve(y_test_cl, y_probs)
plt.plot(fpr, tpr, marker='o')
plt.ylim([0,1.1])
plt.xlim([0,1.1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC curve')
plt.show()
print(f"Площадь под ROC-кривой: {auc(fpr, tpr)}")

# Оценка классификации
report = classification_report(y_test_cl, y_pred)
print(report)
cm = confusion_matrix(y_test_cl, y_pred)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='bwr')
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Решение задачи классификации методом градиентного бустинга
# n_estimators количество слабых моделей
# learning_rate - шаг сходимости

gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=23)
gb_model.fit(X_train_cl, y_train_cl)

y_pred_gb = gb_model.predict(X_test_cl)
y_probs_gb = gb_model.predict_proba(X_test_cl)[:, 1]

# Построения ROC-кривой для классификации методом градиентного бустинга
fpr, tpr, thresholds = roc_curve(y_test_cl, y_probs_gb)
plt.plot(fpr, tpr, marker='o')
plt.ylim([0,1.1])
plt.xlim([0,1.1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC curve')
plt.show()
print(f"Площадь под ROC-кривой: {auc(fpr, tpr)}")

# Оценка классификации
report = classification_report(y_test_cl, y_pred_gb)
print(report)
cm = confusion_matrix(y_test_cl, y_pred_gb)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='bwr')
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()