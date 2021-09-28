import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

"""
В этом задании вам нужно проследить за изменением качества случайного леса в зависимости от количества деревьев в нем.
1. Загрузите данные из файла abalone.csv. 
Это датасет, в котором требуется предсказать возраст ракушки (число колец) по физическим измерениям.
2. Преобразуйте признак Sex в числовой: значение F должно перейти в -1, I — в 0, M — в 1. 
Если вы используете Pandas, то подойдет следующий код: 
    data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
3. Разделите содержимое файлов на признаки и целевую переменную. 
В последнем столбце записана целевая переменная, в остальных — признаки.
4. Обучите случайный лес (sklearn.ensemble.RandomForestRegressor) с различным числом деревьев: от 1 до 50 
(не забудьте выставить "random_state=1" в конструкторе). 
Для каждого из вариантов оцените качество работы полученного леса на кросс-валидации по 5 блокам. 
Используйте параметры "random_state=1" и "shuffle=True" при создании 
генератора кросс-валидации sklearn.cross_validation.KFold.  
В качестве меры качества воспользуйтесь коэффициентом детерминации (sklearn.metrics.r2_score).
5. Определите, при каком минимальном количестве деревьев случайный лес показывает качество на кросс-валидации выше 0.52. 
Это количество и будет ответом на задание.
6. Обратите внимание на изменение качества по мере роста числа деревьев. Ухудшается ли оно?
"""

data = pd.read_csv('static/abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
X = data[['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight']]
y = data['Rings']
kf = KFold(n_splits=5, random_state=1, shuffle=True)
for i in range(1, 51):
    regres = RandomForestRegressor(random_state=1, n_estimators=i, n_jobs=-1)
    quality = cross_val_score(regres, X, y, cv=kf, scoring='r2', n_jobs=-1).mean()
    if round(quality, 3) > 0.52:
        print(i)
        break
