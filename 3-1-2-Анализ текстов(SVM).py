from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import GridSearchCV


newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)

features = newsgroups.data
class_number = newsgroups.target

"""
Одна из сложностей работы с текстовыми данными состоит в том, что для них нужно построить числовое представление.
Одним из способов нахождения такого представления является вычисление TF-IDF.
В Scikit-Learn это реализовано в классе sklearn.feature_extraction.text.TfidfVectorizer.
Преобразование обучающей выборки нужно делать с помощью функции fit_transform,
тестовой — с помощью transform.
"""

vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(features)
kf = KFold(n_splits=5, random_state=241, shuffle=True)

"""
Первым аргументом в GridSearchCV передается классификатор,
для которого будут подбираться значения параметров, вторым — словарь (dict),
задающий сетку параметров для перебора.
После того, как перебор окончен, можно проанализировать значения качества
для всех значений параметров и выбрать наилучший вариант
"""

grid = {'C': np.power(10.0, np.arange(-5, 6))}
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=kf, n_jobs=-1)
gs.fit(features, class_number)
C = gs.best_params_.get('C')
svc = SVC(C=C, random_state=241, kernel='linear')
svc.fit(features, class_number)

"""
Найдите 10 слов с наибольшим абсолютным значением веса (веса хранятся в поле coef_ у svm.SVC).
Они являются ответом на это задание. Укажите эти слова через запятую или пробел,
в нижнем регистре, в лексикографическом порядке.
"""

# Первый вариант работы с матрицей
w = np.argsort(np.abs(np.asarray(svc.coef_.todense())).reshape(-1))[-10:]
myList1 = []
for i in w:
    feature_mapping = vectorizer.get_feature_names_out()
    myList1.append(feature_mapping[i])
print(*sorted(myList1))

# Второй вариант работы с матрицей
coefs = abs(svc.coef_.todense().A1)
coefs = np.argsort(coefs)[-10:]
myList2 = []
for i in w:
    feature_mapping = vectorizer.get_feature_names_out()
    myList2.append(feature_mapping[i])
print(*sorted(myList2))

# Третий вариант работы с матрицей
coefs = abs(svc.coef_.toarray())
coefs = np.argsort(coefs)[0][-10:]
myList3 = []
for i in coefs:
    feature_mapping = vectorizer.get_feature_names_out()
    myList3.append(feature_mapping[i])
print(*sorted(myList3))
