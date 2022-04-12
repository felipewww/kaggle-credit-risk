import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
base_credit = pd.read_csv('./credit_risk_dataset.csv')


# Limpar base dos outliers
base = base_credit[base_credit['person_income'] <= 250000]
plt.hist(x = base['person_income'])

#0 = Paga, 1 = Não paga. Temos muito mais pagantes
sns.countplot(x = base['loan_status'])

# Limpar dados nulos ou números inválidos
base.loc[pd.isnull(base['loan_int_rate']), 'loan_int_rate'] = base['loan_int_rate'].mean()
base.loc[pd.isnull(base['person_emp_length']), 'person_emp_length'] = base['person_emp_length'].mean()

# Previsores
X_credit = base_credit.iloc[:, [0,1,6]] #age, income, loan_amnt
Y_credit = base_credit.iloc[:, 8] # classe 0 ou 1
np_x_credit = X_credit.values
np_y_credit = Y_credit.values

# escalar valores para "income" não ser considerado mais importante que "age" por conta do tamanho do número
creditStandardScaler = StandardScaler()
np_x_credit_fit = creditStandardScaler.fit_transform(np_x_credit)

# separar base de treinamento e de teste
npx_train, npx_test, npy_train, npy_test = train_test_split(np_x_credit_fit, np_y_credit, test_size = 0.1, random_state = 0)

# treinamento algoritimo com base TRAIN
naive_credit = GaussianNB()
naive_credit.fit(npx_train, npy_train)

# previsão com base de TEST
predict = naive_credit.predict(npx_test)

# Validação da eficiência
accuracy_score(npy_test, predict)
confusion_matrix(npy_test, predict)