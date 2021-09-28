import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

"""
1. Загрузите файл classification.csv. В нем записаны истинные классы объектов выборки (колонка true)
и ответы некоторого классификатора (колонка pred).

2. Заполните таблицу ошибок классификации:
Actual Positive
Actual Negative
Predicted Positive
Predicted Negative
Для этого подсчитайте величины TP, FP, FN и TN согласно их определениям. Например, FP — это количество объектов,
имеющих класс 0, но отнесенных алгоритмом к классу 1. Ответ в данном вопросе — четыре числа через пробел.

3. Посчитайте основные метрики качества классификатора:
• Accuracy (доля верно угаданных) — sklearn.metrics.accuracy_score
• Precision (точность) — sklearn.metrics.precision_score
• Recall (полнота) — sklearn.metrics.recall_score
• F-мера — sklearn.metrics.f1_score
"""


def roc_auc(x):
    return round(roc_auc_score(true, x), 2)


def max_precision(x):
    t = precision_recall_curve(true, x)
    precision = []
    for i in range(len(t[1])):
        if t[1][i] >= 0.7:
            precision.append(t[0][i])
    return max(precision)


data = pd.read_csv('static/classification.csv')
true = data['true']
pred = data['pred']
TP = 0
FP = 0
TN = 0
FN = 0
for i in data.values:
    if i[0] == 0 and i[0] == i[1]:
        TN += 1
    elif i[0] == 1 and i[0] == i[1]:
        TP += 1
    elif i[0] == 1 and i[0] != i[1]:
        FN += 1
    elif i[0] == 0 and i[0] != i[1]:
        FP += 1
print(TP, FP, FN, TN)
print(
    *map(
        lambda x: round(x, 2),
        [accuracy_score(true, pred), precision_score(true, pred), recall_score(true, pred), f1_score(true, pred)]
    )
)

"""
4. Имеется четыре обученных классификатора. В файле scores.csv записаны истинные классы и значения степени
принадлежности положительному классу для каждого классификатора на некоторой выборке:
• для логистической регрессии — вероятность положительного класса (колонка score_logreg),
• для SVM — отступ от разделяющей поверхности (колонка score_svm),
• для метрического алгоритма — взвешенная сумма классов соседей (колонка score_knn),
• для решающего дерева — доля положительных объектов в листе (колонка score_tree).
Загрузите этот файл.

5. Посчитайте площадь под ROC-кривой для каждого классификатора. Какой классификатор имеет наибольшее значение
метрики AUC-ROC (укажите название столбца)? Воспользуйтесь функцией sklearn.metrics.roc_auc_score.
"""

data = pd.read_csv('static/scores.csv')
true = data['true']
logreg = data['score_logreg']
SVM = data['score_svm']
metr_algo = data['score_knn']
tree = data['score_tree']
myList = [logreg, SVM, metr_algo, tree]
score = max(map(lambda x: roc_auc(x), myList))
for i in myList:
    if roc_auc(i) == score:
        print(i.name)

"""
6. Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70% ?
Чтобы получить ответ на этот вопрос, найдите все точки precision-recall-кривой с помощью функции
sklearn.metrics.precision_recall_curve. Она возвращает три массива: precision, recall, thresholds.
В них записаны точность и полнота при определенных порогах, указанных в массиве thresholds.
Найдите максимальной значение точности среди тех записей, для которых полнота не меньше, чем 0.7.
"""

max_score = max(map(lambda x: max_precision(x), myList))
for i in myList:
    if max_precision(i) == max_score:
        print(i.name)
