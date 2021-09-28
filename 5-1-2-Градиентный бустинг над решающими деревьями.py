import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

"""
  Загрузите выборку из файла gbm-data.csv с помощью pandas и преобразуйте ее в массив numpy
(параметр values у датафрейма). В первой колонке файла с данными записано, была или нет реакция.
  Все остальные колонки (d1 - d1776) содержат различные характеристики молекулы, такие как размер,
форма и т.д. Разбейте выборку на обучающую и тестовую, используя функцию train_test_split
с параметрами test_size = 0.8 и random_state = 241.
  Обучите GradientBoostingClassifier с параметрами n_estimators=250, verbose=True, random_state=241 и
для каждого значения learning_rate из списка [1, 0.5, 0.3, 0.2, 0.1] проделайте следующее:
Используйте метод staged_decision_function для предсказания качества на обучающей и тестовой выборке
на каждой итерации.
  Преобразуйте полученное предсказание с помощью сигмоидной функции по формуле 1 / (1 + e^{−y_pred}),
где y_pred — предсказанное значение.
  Вычислите и постройте график значений log-loss (которую можно посчитать с помощью функции sklearn.metrics.log_loss)
"""

data = pd.read_csv('static/gbm-data.csv')
data = data.values
y = data[:, 0]
X = data[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)


learning_rate = [1, 0.5, 0.3, 0.2, 0.1]
for i in range(len(learning_rate)):
    test_loss = list()
    train_loss = list()
    clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=learning_rate[i])
    clf.fit(X_train, y_train)
    for y_pred in clf.staged_decision_function(X_test):
        y_pred = 1.0 / (1.0 + np.exp(- y_pred))
        test_loss.append(log_loss(y_test, y_pred))
    for y_pred in clf.staged_decision_function(X_train):
        y_pred = 1.0 / (1.0 + np.exp(- y_pred))
        train_loss.append(log_loss(y_train, y_pred))
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.show()

"""
4. Приведите минимальное значение log-loss на тестовой выборке и номер итерации, на котором оно достигается,
при learning_rate = 0.2.
"""

clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=0.2)
clf.fit(X_train, y_train)
test_loss = list()
for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
    y_pred = 1.0 / (1.0 + np.exp(- y_pred))
    test_loss.append([i + 1, log_loss(y_test, y_pred)])
test_loss = pd.DataFrame(test_loss, columns=['iter', 'loss'])
print(test_loss[test_loss.loss == test_loss.loss.min()])

"""
5. На этих же данных обучите RandomForestClassifier с количеством деревьев, равным количеству итераций,
на котором достигается наилучшее качество у градиентного бустинга из предыдущего пункта, c random_state=241 и
остальными параметрами по умолчанию. Какое значение log-loss на тесте получается у этого случайного леса?
(Не забывайте, что предсказания нужно получать с помощью функции predict_proba. В данном случае брать сигмоиду
от оценки вероятности класса не нужно)
"""

test_loss = list()
clf_tree = RandomForestClassifier(random_state=1, n_estimators=37, n_jobs=-1)
clf_tree.fit(X_train, y_train)
print(log_loss(y_test, clf_tree.predict_proba(X_test)))
