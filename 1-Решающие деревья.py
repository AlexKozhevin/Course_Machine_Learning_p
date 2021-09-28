import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# X = np.array([[1, 2], [3, 4], [5, 6]])
# y = np.array([0, 1, 0])
# clf = DecisionTreeClassifier()
# clf.fit(X, y)
# clf = DecisionTreeClassifier()

"""
1. Загрузите выборку из файла titanic.csv с помощью пакета Pandas.

2. Оставьте в выборке четыре признака: класс пассажира (Pclass), цену билета (Fare),
возраст пассажира (Age) и его пол (Sex).

3. Обратите внимание, что признак Sex имеет строковые значения.

4. Выделите целевую переменную — она записана в столбце Survived.

5. В данных есть пропущенные значения — например, для некоторых пассажиров неизвестен их возраст.
Такие записи при чтении их в pandas принимают значение nan.
Найдите все объекты, у которых есть пропущенные признаки, и удалите их из выборки.

6. Обучите решающее дерево с параметром random_state=241 и остальными параметрами по умолчанию
(речь идет о параметрах конструктора DecisionTreeСlassifier).

7. Вычислите важности признаков и найдите два признака с наибольшей важностью.
Их названия будут ответами для данной задачи
(в качестве ответа укажите названия признаков через запятую или пробел, порядок не важен).
"""

data = pd.read_csv('static/titanic.csv')
data_fin = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]
data_fin = data_fin.dropna(axis='index', how='any')
survived = data_fin['Survived']
data_fin = data_fin[['Pclass', 'Fare', 'Age', 'Sex']]
data_fin.loc[(data_fin.Sex == 'female'), 'Sex'] = '0'
data_fin.loc[(data_fin.Sex == 'male'), 'Sex'] = '1'
clf = DecisionTreeClassifier(random_state=241)
clf.fit(data_fin, survived)
importances = clf.feature_importances_
print(*importances)
print(*data_fin)
