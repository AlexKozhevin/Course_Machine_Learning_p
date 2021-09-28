import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data_train_read = pd.read_csv('static/perceptron-train.csv', header=None)  # !!!!!!!!! header=None !!!!!!!
data_test_read = pd.read_csv('static/perceptron-test.csv', header=None)    # !!!!!!!!! header=None !!!!!!!
target_train = data_train_read.iloc[:, 0]
data_train = data_train_read.iloc[:, 1:3]
target_test = data_test_read.iloc[:, 0]
data_test = data_test_read.iloc[:, 1:3]

clf1 = Perceptron(random_state=241)
clf1.fit(data_train, target_train)  # получили 'метод' для выборки
predictions = clf1.predict(data_test)   # проверили метод на тестовых данных

quality1 = accuracy_score(target_test, predictions)  # посчитали качество используя тестовые ответы и проверку метода
print(quality1)

scaler = StandardScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)

clf2 = Perceptron()  # так как признаки нормализованы, не используем random_state
clf2.fit(data_train, target_train)
predictions = clf2.predict(data_test)

quality2 = accuracy_score(target_test, predictions)
print(quality2)

print(round((quality2 - quality1), 3))
