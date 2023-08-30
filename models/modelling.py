import pandas as pd
from pathlib import Path
import joblib

df = pd.read_csv('cleaned.csv')

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

with open('model.pkl', 'wb') as file:
  joblib.dump(bst, file, compress=('gzip', 3))
