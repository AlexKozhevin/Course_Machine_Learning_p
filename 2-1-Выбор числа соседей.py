import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier

"""
Для начала нужно разделить данные
Далее, чтобы работала функция cross_val_score() необходимо определить KNeighborsClassifier() и KFold()

KFold() - определяет как будет проводиться разбивка признаков, сообщить в параметры нужно n_splits=5,
shuffle=True, random_state=42, как указано в задании, и все. Зачем некоторые люди добавляют туда len(y)
я совсем не понимаю, возможно был апдейт sklearn после которого все поменялось.

KNeighborsClassifier() - с ним уже интереснее, в параметр нужно сообщить число соседей n_neighbors=k и
все. k надо прогнать от 1 до 50(51). Значит, что KNeighborsClassifier(n_neighbors=k) у нас будет в цикле.
Классификатор не нужно обучать( делать KNeighborsClassifier().fit(x,y) ), потому что cross_val_score()
делает это сама.

cross_val_score() - ключевая функция в данном задании, выдаст вам 5 значений точности кросс-валидации,
которые нужно будет усреднить, я использовал np.mean(). В параметры сообщаем следующее

quality = cross_val_score(classifier, x, y, cv=kf, scoring='accuracy') - наш оценщик

classifie - так я обозначил KNeighborsClassifier(n_neighbors=k)

cv = KFold(n_splits=5, shuffle=True, random_state=42) - генератор разбиений

cross_val_score тоже будет в цикле и будет оценивать классификатор при значениях от 1 до 50.
Все усредненные значения кроса надо будет сравнить, запомнить наибольшее среднее значение оценки точности
и количество соседей, при котором оно существует.

Потом применяете масштабирование scale только к признакам x и повторяете цикл.
"""

features = pd.read_csv('static/wine.data', index_col='class')
class_wine = list(features.index)
kf = KFold(n_splits=5, random_state=42, shuffle=True)
cross_val1 = []
for k in range(1, 51):
    classifier = KNeighborsClassifier(n_neighbors=k)
    quality = cross_val_score(classifier, features, class_wine, cv=kf, scoring='accuracy').mean()
    cross_val1.append(quality)
my_dict1 = dict(zip(cross_val1, range(1, 51)))
print(my_dict1[max(cross_val1)])
print(round(max(cross_val1), 2))

features = scale(features)
cross_val2 = []
for k in range(1, 51):
    classifier = KNeighborsClassifier(n_neighbors=k)
    quality = cross_val_score(classifier, features, class_wine, cv=kf, scoring='accuracy').mean()
    cross_val2.append(quality)
my_dict2 = dict(zip(cross_val2, range(1, 51)))
print(my_dict2[max(cross_val2)])
print(round(max(cross_val2), 2))
