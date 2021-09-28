from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

"""
1. Загрузите данные об описаниях вакансий и соответствующих годовых зарплатах из файла salary-train.csv 
(либо его заархивированную версию salary-train.zip).
2. Проведите предобработку:
- Приведите тексты к нижнему регистру (text.lower()).
- Замените все, кроме букв и цифр, на пробелы — это облегчит дальнейшее разделение текста на слова. 
Для такой замены в строке text подходит следующий вызов: re.sub('[^a-zA-Z0-9]', ' ', text). 
Также можно воспользоваться методом replace у DataFrame, чтобы сразу преобразовать все тексты:

    train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

- Примените TfidfVectorizer для преобразования текстов в векторы признаков. Оставьте только те слова, 
которые встречаются хотя бы в 5 объектах (параметр min_df у TfidfVectorizer).
- Замените пропуски в столбцах LocationNormalized и ContractTime на специальную строку 'nan'. 
Код для этого был приведен выше.
- Примените DictVectorizer для получения one-hot-кодирования признаков LocationNormalized и ContractTime.
- Объедините все полученные признаки в одну матрицу "объекты-признаки". Обратите внимание, 
что матрицы для текстов и категориальных признаков являются разреженными. 
Для объединения их столбцов нужно воспользоваться функцией scipy.sparse.hstack.
3.  Обучите гребневую регрессию с параметрами alpha=1 и random_state=241. 
Целевая переменная записана в столбце SalaryNormalized.
4.  Постройте прогнозы для двух примеров из файла salary-test-mini.csv. 
Значения полученных прогнозов являются ответом на задание. Укажите их через пробел.
"""

data_train = pd.read_csv('static/salary-train.csv')
data_test = pd.read_csv('static/salary-test-mini.csv')
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)
data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)
text_train = data_train['FullDescription']
text_test = data_test['FullDescription']
text_train = text_train.str.lower()
text_test = text_test.str.lower()
text_train = text_train.replace('[^a-zA-Z0-9]', ' ', regex=True)
text_test = text_test.replace('[^a-zA-Z0-9]', ' ', regex=True)

vectorizer = TfidfVectorizer(min_df=5)
text_train = vectorizer.fit_transform(text_train)
text_test = vectorizer.transform(text_test)

enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_train = hstack([text_train, X_train_categ])
X_test = hstack([text_test, X_test_categ])
Y_train = data_train['SalaryNormalized']

clf = Ridge(alpha=1, random_state=241)
clf.fit(X_train, Y_train)
for i in clf.predict(X_test):
    print(round(i, 2))
