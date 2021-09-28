import numpy as np

"""
Выведите для заданной матрицы номера строк, сумма элементов в которых превосходит 10.
Функция для подсчета суммы: np.sum
Аргументы аналогичны функциям np.mean и np.std.
К матрицам можно применять логические операции, которые будут применяться поэлементно. Соответственно,
результатом такой операции будет матрица такого же размера, в ячейках которой будет записано либо True, либо False.
Индексы элементов со значением True можно получить с помощью функции np.nonzero.
"""

Z = np.array([[4, 5, 0],
              [1, 9, 3],
              [5, 1, 1],
              [3, 3, 3],
              [9, 9, 9],
              [4, 7, 1]])
r = np.sum(Z, axis=1)
for r in np.nonzero(r > 10):
    print(*r, sep=', ')
