import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

url = 'dataset/diabetes.csv'
diabetes = pd.read_csv(url)

import pandas as pd

# Misalnya, jika Anda memiliki DataFrame 'data':
# Ganti nilai 0 dengan NaN pada kolom yang memiliki nilai kosong
columns_with_missing_values = ['BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Glucose']
diabetes[columns_with_missing_values] = diabetes[columns_with_missing_values].replace(0, pd.NA)

# Sekarang, gantilah nilai NaN dengan mean pada masing-masing kolom
for column in columns_with_missing_values:
    diabetes[column].fillna(diabetes[column].mean(), inplace=True)
    

# Misalnya, jika Anda memiliki DataFrame 'data' dan ingin menggantikan nilai missing di kolom 'BloodPressure' dengan mean:
diabetes['BloodPressure'].fillna(diabetes['BloodPressure'].mean(), inplace=True)

# Untuk menggantikan nilai missing di kolom 'SkinThickness' dengan mean:
diabetes['SkinThickness'].fillna(diabetes['SkinThickness'].mean(), inplace=True)

# Menggantikan nilai missing di kolom 'Insulin' dengan mean:
diabetes['Insulin'].fillna(diabetes['Insulin'].mean(), inplace=True)

# Menggantikan nilai missing di kolom 'BMI' dengan mean:
diabetes['BMI'].fillna(diabetes['BMI'].mean(), inplace=True)

# Menggantikan nilai missing di kolom 'Glucose' dengan mean:
diabetes['Glucose'].fillna(diabetes['BMI'].mean(), inplace=True)

from sklearn.model_selection import train_test_split
 
X = diabetes.drop(["Outcome"],axis =1)
y = diabetes["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123)

from sklearn.preprocessing import StandardScaler
import pandas as pd

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Siapkan dataframe untuk analisis model
models = pd.DataFrame(index=['train_mse', 'test_mse'], 
                      columns=['KNN'])

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
 
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)
 
models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)

# Buat variabel mse yang isinya adalah dataframe nilai mse data train dan test pada masing-masing algoritma
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN'])
 
# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN': knn}
 
# Hitung Mean Squared Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3 
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3
