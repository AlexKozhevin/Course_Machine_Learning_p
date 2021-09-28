import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

"""
1. Загрузите данные close_prices.csv. В этом файле приведены цены акций 30 компаний
на закрытии торгов за каждый день периода.
2. На загруженных данных обучите преобразование PCA с числом компоненты равным 10.
Скольких компонент хватит, чтобы объяснить 90% дисперсии?
"""

data = pd.read_csv('static/close_prices.csv', index_col='date')
dow = pd.read_csv('static/djia_index.csv', index_col='date')
pca = PCA(n_components=10)
pca.fit(data)

myList1 = sorted(pca.explained_variance_ratio_)
myList1.reverse()
summ = 0
k = 0
for i in range(len(myList1)):
    if summ < 0.9:
        summ = summ + myList1[i]
        myList1[i] += myList1[i + 1]
        k += 1
    else:
        break
print(k)

"""
3. Примените построенное преобразование к исходным данным и возьмите значения первой компоненты.
Загрузите информацию об индексе Доу-Джонса из файла djia_index.csv.
4. Чему равна корреляция Пирсона между первой компонентой и индексом Доу-Джонса?
"""

X = pca.transform(data)
print(round(np.corrcoef(X[:, 0], dow['^DJI'])[0][1], 2))

"""
5. Какая компания имеет наибольший вес в первой компоненте? Укажите ее название с большой буквы.
"""

names = list(data.iloc[0].index)
my_dict1 = dict(zip(pca.components_[0], names))
print(my_dict1[max(pca.components_[0])])
# print(round(max(pca.components_[0]), 2))
