import pandas as pd
from sklearn.svm import SVC

data = pd.read_csv('static/svm-data.csv', header=None)  # !!!!!!!!! header=None !!!!!!!
aim = data.iloc[:, 0]
features = data.iloc[:, 1:]
svc = SVC(C=100000, random_state=241, kernel='linear')
svc.fit(features, aim)
print(*map(lambda x: x + 1, svc.support_))
