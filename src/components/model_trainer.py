import pandas as pd
import joblib

df = pd.read_csv('../artifact/Data.csv')
df['Class'] = df['Class'].map(lambda x: 1 if x == 2 else 0)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=0)

from xgboost import XGBClassifier

bst = XGBClassifier()
bst.fit(X_train, y_train)

with open('../artifact/model.pkl', 'wb') as file:
  joblib.dump(bst, file, compress=('gzip', 3))
