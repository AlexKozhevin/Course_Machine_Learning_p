import pandas
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import datetime

"""
Про задачу и финальное задание
Почему именно такая задача?
Публикация реальных данных из индустриальных задач — очень смелый шаг для компании. Мало кто (даже Яндекс) 
может на такое пойти. Гораздо проще (а порой и интереснее) воспользоваться данными из открытых источников.
Публичные датасеты из интернета для решения реальных бизнес-задач мало пригодны, собственно поэтому они и лежат 
в открытом доступе.
Мы предпочли сделать игрушечную задачу на реальных данных, вместо реальной задачи на игрушечных данных.
Задача прогнозирования победы — игрушечная, но вот лишь небольшой перечень реальных задач, на которые она похожа:
предсказания вероятности покупки услуги клиентом банка
предсказание вероятности оттока клиента к другому поставщику услуг
... (подумайте над другими примерами)
Задание слишком простое. Что еще можно сделать?
Ответить на вопрос: какое минимальное число минут матча необходимо знать, для того чтобы в 80% матчей верно угадывать 
победившую сторону? А с точностью 90%? Дайте свой ответ на этот вопрос и докажите что такой точности
действительно можно достичь, построив модель и качественно провалидировав ее. 
Насколько матчи в игре Dota 2 предсказуемы?

Напишите об этом статью, расскажите всем, и приходите к нам на собеседование.

Где взяли данные?
Набор данных был сделан на основе выгрузки YASP 3.5 Million Data Dump реплеев матчей Dota 2 с сайта yasp.co. 
За выгрузку огромное спасибо Albert Cui and Howard Chung and Nicholas Hanson-Holtry. Лицензия на выгрузку: CC BY-SA 4.0.

Как сформировали выборку?
Оригинальная выгрузка матчей была очищена, в предложенном наборе присутствуют матчи:

сыгранные с 2015-05-01 до 2015-12-17
длительностью не менее 15 минут
убраны матчи с неполной информацией (например: отсутвует информация про игроков)
Из всего датасета 15% случайных записей были выделены в тестовое множество.

Для того чтобы размотивировать участников соревнования на Kaggle занимать высокие места читерскими методами 
(например, скачав оригинальный набор данных и подсмотрев ответы на тестовом множестве матчей), 
мы произвели минимальную обфускацию данных, т.е. немного запутали датасет:
поменяли идентификаторы матчей
время начала каждого матча сдвинули на значение случайной величины, нормально распределенной со стандартным отклонением 
в 1 сутки
"""

data = pandas.read_csv('static/features.csv', index_col='match_id')
data.head()
data_test = pandas.read_csv('static/features_test.csv', index_col='match_id')
data_test.head()
data_test = data_test.fillna(0)
features = data.drop(["duration", "radiant_win", "tower_status_radiant", "tower_status_dire",
                      "barracks_status_dire", "barracks_status_radiant"], axis=1)

"""
Проверьте выборку на наличие пропусков с помощью функции count(), которая для каждого столбца показывает
число заполненных значений. Много ли пропусков в данных? Запишите названия признаков, имеющих пропуски
"""
myList = []
for i in range(len(features.count())):
    if len(features) > features.count()[i]:
        myList.append(features.count().index[i])
print("Названия признаков:")
print(*myList, sep=', ')

"""
Замените пропуски на нули с помощью функции fillna()
"""
X = features = features.fillna(0)
y = data["radiant_win"]

"""
Оцените качество градиентного бустинга (GradientBoostingClassifier) с помощью данной кросс-валидации
"""
Log_X = preprocessing.StandardScaler().fit_transform(X)
start_time = datetime.datetime.now()
kf = KFold(n_splits=5, random_state=1, shuffle=True)
clf = GradientBoostingClassifier(n_estimators=30, verbose=True, random_state=241, learning_rate=0.5, max_depth=3)
quality = cross_val_score(clf, Log_X, y, cv=kf, scoring='roc_auc', n_jobs=-1).mean()
print('Time elapsed:', datetime.datetime.now() - start_time)
print("Качество градиентного бустинга:", quality)

"""
Оцените качество логистической регрессии (sklearn.linear_model.LogisticRegression с L2-регуляризацией)
с помощью кросс-валидации по той же схеме, которая использовалась для градиентного бустинга.
Подберите при этом лучший параметр регуляризации (C).
"""
Log_X = preprocessing.StandardScaler().fit_transform(X)
start_time = datetime.datetime.now()
grid = {'C': np.power(10.0, np.arange(-5, 6))}
clf = LogisticRegression(random_state=241, n_jobs=-1)
gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=kf, n_jobs=-1)
gs.fit(Log_X, y)
C = gs.best_params_.get('C')
clf = LogisticRegression(C=C, random_state=241, n_jobs=-1)
quality = cross_val_score(clf, Log_X, y, cv=kf, scoring='roc_auc', n_jobs=-1).mean()
print('Time elapsed:', datetime.datetime.now() - start_time)
print("Качество логистической регрессии:", quality)

"""
Уберите категориальные признаки из выборки, и проведите кросс-валидацию для логистической регрессии на новой выборке
с подбором лучшего параметра регуляризации.
"""
X_no_hero = X.copy()
X_only_hero = pandas.DataFrame()
X_no_hero = X_no_hero.drop(columns=['lobby_type'])

for i in range(1, 6):
    X_only_hero = pandas.concat([X_only_hero, X[[f'r{i}_hero', f'd{i}_hero']]], axis=1)
    X_no_hero = X_no_hero.drop(columns=[f'r{i}_hero', f'd{i}_hero'])

X_no_hero_test = data_test.copy()
X_only_hero_test = pandas.DataFrame()
X_no_hero_test = X_no_hero_test.drop(columns=['lobby_type'])

for i in range(1, 6):
    X_only_hero_test = pandas.concat([X_only_hero_test, data_test[[f'r{i}_hero', f'd{i}_hero']]], axis=1)
    X_no_hero_test = X_no_hero_test.drop(columns=[f'r{i}_hero', f'd{i}_hero'])

Log_X = preprocessing.StandardScaler().fit_transform(X_no_hero)
start_time = datetime.datetime.now()
grid = {'C': np.power(10.0, np.arange(-5, 6))}
clf = LogisticRegression(random_state=241, n_jobs=-1)
gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=kf, n_jobs=-1)
gs.fit(Log_X, y)
C = gs.best_params_.get('C')
clf = LogisticRegression(C=C, random_state=241, n_jobs=-1)
quality = cross_val_score(clf, Log_X, y, cv=kf, scoring='roc_auc', n_jobs=-1).mean()
print('Time elapsed:', datetime.datetime.now() - start_time)
print("Качество логистической регрессии на новой выборке с подбором лучшего параметра регуляризации:", quality)

"""
Выясните из данных, сколько различных идентификаторов героев существует в данной игре
"""
N_max = max(X_only_hero.values.reshape(-1))
print(f'Итого {N_max} идентификаторов героев')

"""
Воспользуемся подходом "мешок слов" для кодирования информации о героях.
"""
X_pick = np.zeros((X_only_hero.shape[0], N_max))
for i, match_id in enumerate(X_only_hero.index):
    for p in range(5):
        X_pick[i, X_only_hero.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, X_only_hero.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1

X_pick = pandas.DataFrame(X_pick, index=X.index)
X_pick = pandas.concat([X_no_hero, X_pick[0:]], axis=1)
# print(X_pick)

X_pick_test = np.zeros((X_only_hero_test.shape[0], N_max))
for i, match_id in enumerate(X_only_hero_test.index):
    for p in range(5):
        X_pick_test[i, X_only_hero_test.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick_test[i, X_only_hero_test.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1
X_pick_test = pandas.DataFrame(X_pick_test, index=data_test.index)
X_pick_test = pandas.concat([X_no_hero_test, X_pick_test[0:]], axis=1)
# print(X_pick_test)

"""
Проведите кросс-валидацию для логистической регрессии на новой выборке с подбором лучшего параметра регуляризации.
"""
Log_X = preprocessing.StandardScaler().fit_transform(X_pick)
start_time = datetime.datetime.now()
grid = {'C': np.power(10.0, np.arange(-5, 6))}
clf = LogisticRegression(random_state=241, n_jobs=-1)
gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=kf, n_jobs=-1)
gs.fit(Log_X, y)
C = gs.best_params_.get('C')
clf = LogisticRegression(C=C, random_state=241, n_jobs=-1)
quality = cross_val_score(clf, Log_X, y, cv=kf, scoring='roc_auc', n_jobs=-1).mean()
print('Time elapsed:', datetime.datetime.now() - start_time)
print("Качество логистической регрессии на новой выборке с подбором лучшего параметра регуляризации:", quality)

"""
Постройте предсказания вероятностей победы команды Radiant для тестовой выборки с помощью лучшей из изученных моделей
"""
clf.fit(Log_X, y)
Log_X_test = preprocessing.StandardScaler().fit_transform(X_pick_test)
pred = clf.predict_proba(Log_X_test)

"""
Какое минимальное и максимальное значение прогноза на тестовой выборке получилось у лучшего из алгоритмов?
"""
print("Минимальное значение прогноза на тестовой выборке:", min(pred[:, 1]))
print("Максимальное значение прогноза на тестовой выборке:", max(pred[:, 1]))

"""
1. Качество при логистической регрессии получилось 0.71637, что больше чем при градиентном бустинге. 
Также расчет происходит значительно быстрее (менее 2-х секунд), без учета времени подбора лучшего параметра 
регуляризации (C). С учетом времени подбора (C), время логистической регрессии почти равно времени градиентного 
бустинга (28,5сек).
2. Удаление категориальных признаков увеличило качество до 0.71641. Категориальные признаки, нежелательно использовать 
как числовые без дополнительных преобразований.
3. 112 различных идентификаторов героев существует в данной игре
4. Качество при добавлении "мешка слов" по героям получилось 0.7519. Качество улучшилось, так как герои имеют разные 
характеристики, и некоторые из них выигрывают чаще, чем другие
5. Минимальное значение 0.0087. Максимальное значение 0.9963.
"""