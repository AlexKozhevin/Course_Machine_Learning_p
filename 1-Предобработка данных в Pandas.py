import pandas
import numpy as np
from statistics import median

# data = pandas.read_csv('titanic.csv', index_col='PassengerId')
data = pandas.read_csv('titanic.csv')
# print(*data.head())
# 1
# print(data['Sex'].value_counts())

# 2
# print(data['Survived'].value_counts())
not_survived, survived = data['Survived'].value_counts()
all_passengers = survived + not_survived
# print(survived * 100 / all_passengers)

# 3
# print(data['Pclass'].value_counts())
# third_cl, first_cl, second_cl = data['Pclass'].value_counts()
# print(round((first_cl * 100 / all_passengers), 2))

# 4
# list_ages = []
# for age in data['Age']:
#     if np.isnan(age) != True:
#         list_ages.append(age)
# print(round(np.mean(list_ages), 2))
# print(round(np.median(list_ages), 2))

# 5
# sib_sp = data['SibSp']
# parch = data['Parch']
# print(round(sib_sp.corr(parch), 2))

# 6
list_female_names = []
for man in data.values:
    if 'female' in list(man):
        list_female_names.append(man[3])
text = str(' '.join(list_female_names).strip())
text = text.replace('(', '')
text = text.replace(')', '')
text = text.replace('.', '')
text = text.replace('"', '')
text = text.replace(',', '')
text = text.replace('Mrs', '')
text = text.replace('Miss', '')
list_female_names = text.split()
# print(list_female_names)

countDict = dict()
for name in list_female_names:
    words = sorted(name.split())
    for word in words:
        countDict[word] = countDict.get(word, 0) + 1
b = max(countDict.values())
for word in sorted(countDict):
    if b == countDict[word]:
        print(word)
