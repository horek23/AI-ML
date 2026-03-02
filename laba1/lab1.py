import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("train.csv")

print("ИНФОРМАЦИЯ О ДАТАСЕТЕ\n")
print(df.info())

print("ДАННЫЕ ИЗ ДАТАСЕТА\n")
print(df.head())

print("КОЛИЧЕСТВО ПРОПУЩЕННЫХ ЗНАЧЕНИЙ ДЛЯ СТОЛБЦОВ, В КОТОРЫХ ЕСТЬ ПРОПУСКИ\n")
nan_matrix = df.isnull()
missing = nan_matrix.sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

print("ЗАПОЛНЕНИЕ ПРОПУСКОВ\n")
#там где много пропусков, заменим пропуск на "None", а также там, где отсутствие признака не случайность, а данные

categorical_none = [ 'PoolQC', 'MiscFeature', 'Alley', 'Fence','MasVnrType', 'FireplaceQu', 'GarageType',
     'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtExposure', 'BsmtQual', 'BsmtCond', 'BsmtFinType1',
     'BsmtFinType2']

for col in categorical_none:
    if col in df.columns:
        df[col] = df[col].fillna("None")

#для категориальных данных, которые должны быть у всех используем моду
Electrical_mode = df['Electrical'].mode()[0]
df['Electrical'] = df['Electrical'].fillna(Electrical_mode)

#если строк <5% от общего количества, то можно просто удалить
df = df.dropna(subset=['MasVnrArea'])

"""
описательная статистика столбца df['LotFrontage'].describe()
(если медиана приблизительно = среднему или 75% не сильно отличается от max, то можно использовать для замены среднее, 
а если сильно отличается, то будем использовать медиану)
Для LotFrontage max сильно отличается от 75%, значит используем медиану
Для GarageYrBlt max не сильно отличается от 75%, значит используем среднее
"""

LotFrontage_median = df['LotFrontage'].median()
df['LotFrontage'] = df['LotFrontage'].fillna(LotFrontage_median)

GarageYrBlt_mean = df['GarageYrBlt'].mean()
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(GarageYrBlt_mean)

nan_matrix = df.isnull()
missing = nan_matrix.sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

print("НОРМАЛИЗАЦИЯ ДАННЫХ\n")

#выбрали Z-оценку, тк она более устойчива к выбросам
#только 0 и 1: BsmtHalfBath, HalfBath, только 1: KitchenAbvGr, ID - их не нужно нормализовать, поэтому исключим
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
numeric_cols = numeric_cols.drop(['Id','BsmtHalfBath','HalfBath','KitchenAbvGr'])
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

#проверяем, что нормализация сработала (среднее должно быть 0, а стандартное отклонение 1)
print("среднее")
print(df[numeric_cols].mean())
print("стандартное отклонение")
print(df[numeric_cols].std())

print("ПРЕОБРАЗОВАНИЕ КАТЕГОРИАЛЬНЫХ ДАННЫХ В ЧИСЛЕННЫЕ\n")
#для порядковых столбцов выберем кодирование последовательностью чисел
#для остальных используем One-hot encoding (OHE)

quality_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5, "None": 0}
cols = ['ExterQual', 'ExterCond', 'BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual',
        'GarageCond','PoolQC',]
for col in cols:
    df[col] = df[col].map(quality_map)

quality_map_1 = {'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4, "None": 0}
df['BsmtExposure'] = df['BsmtExposure'].map(quality_map_1)

quality_map_2 = {'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6, "None": 0}
df['BsmtFinType1'] = df['BsmtFinType1'].map(quality_map_2)
df['BsmtFinType2'] = df['BsmtFinType2'].map(quality_map_2)

quality_map_3 = {'Unf': 1, 'RFn': 2, 'Fin': 3, "None": 0}
df['GarageFinish'] = df['GarageFinish'].map(quality_map_3)

quality_map_4 = {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8, "None": 0}
df['Functional'] = df['Functional'].map(quality_map_4)

quality_map_5 = {'Sev': 1, 'Mod': 2, 'Gtl': 3, "None": 0}
df['LandSlope'] = df['LandSlope'].map(quality_map_5)

quality_map_6 = {'IR3': 1, 'IR2': 2, 'IR1': 3, "Reg": 4, "None": 0}
df['LotShape'] = df['LotShape'].map(quality_map_6)

quality_map_7 = {'N': 1, 'P': 2, 'Y': 3, "None": 0}
df['PavedDrive'] = df['PavedDrive'].map(quality_map_7)

nominal_cols = ['MSZoning','Street','Alley','LandContour','Utilities','LotConfig','Neighborhood','Condition1',
    'Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
    'Foundation','Heating','CentralAir','Electrical','GarageType','MiscFeature','SaleType','SaleCondition']

df = pd.get_dummies(df, columns = nominal_cols, drop_first=True)

nan_matrix = df.isnull()
missing = nan_matrix.sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

df.to_csv("clean_dataset.csv", index=False)
