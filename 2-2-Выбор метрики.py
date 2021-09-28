import sklearn
import numpy
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

"""
1. Загрузите выборку Boston с помощью функции sklearn.datasets.load_boston(). Результатом вызова данной функции
является объект, у которого признаки записаны в поле data, а целевой вектор — в поле target.

2. Приведите признаки в выборке к одному масштабу при помощи функции sklearn.preprocessing.scale.

3. Переберите разные варианты параметра метрики p по сетке от 1 до 10 с таким шагом, чтобы всего было
протестировано 200 вариантов (используйте функцию numpy.linspace).
Используйте KNeighborsRegressor с n_neighbors=5 и weights='distance' — данный параметр добавляет в алгоритм веса,
зависящие от расстояния до ближайших соседей. В качестве метрики качества используйте среднеквадратичную ошибку
(параметр scoring='mean_squared_error' у cross_val_score; при использовании
библиотеки scikit-learn версии 0.18.1 и выше необходимо указывать scoring='neg_mean_squared_error').
Качество оценивайте, как и в предыдущем задании, с помощью кросс-валидации по 5 блокам с random_state = 42,
не забудьте включить перемешивание выборки (shuffle=True).

4. Определите, при каком p качество на кросс-валидации оказалось оптимальным. Обратите внимание,
что cross_val_score возвращает массив показателей качества по блокам; необходимо максимизировать среднее
этих показателей. Это значение параметра и будет ответом на задачу.
"""

data, target = load_boston(return_X_y=True)
data_scale = sklearn.preprocessing.scale(data)
x1 = numpy.linspace(1, 10, num=200)
kf = KFold(n_splits=5, random_state=42, shuffle=True)
cross_val1 = []
for x in x1:
    regression = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=x)
    quality = cross_val_score(regression, data, target, cv=kf, scoring='neg_mean_squared_error').mean()
    cross_val1.append(quality)
my_dict1 = dict(zip(cross_val1, range(0, 199)))
print(x1[my_dict1[max(cross_val1)]])
