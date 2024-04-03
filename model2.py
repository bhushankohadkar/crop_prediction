import pickle
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score 


df = pd.read_csv(r"FinalData.csv")

features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall','soil_moisture']]
target = df['label']

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state=2)

le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)  # Use transform instead of fit_transform for test data

dt = DecisionTreeClassifier()

scores = cross_val_score(dt, X_train, Y_train, cv=10, verbose=3, n_jobs=1)
print(scores.mean())

param_dict={"max_depth":range(1,100)}
grid=RandomizedSearchCV(dt,param_dict,cv=5)
grid.fit(X_train,Y_train)
print(grid.best_score_)
print(grid.best_estimator_)
print(grid.best_params_)

dt=DecisionTreeClassifier(max_depth=19)
dt.fit(X_train,Y_train)

Y_pred_encoded = dt.predict(X_test)

# Inverse transform the predicted and actual labels
Y_pred_original = le.inverse_transform(Y_pred_encoded)

# Evaluate model
print("Accuracy:", accuracy_score(Y_test, Y_pred_original))
pickle.dump(dt, open('model2.pkl', 'wb'))
pickle.dump(le, open('label.pkl', 'wb'))