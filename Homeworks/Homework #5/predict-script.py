import pickle
import os

dir = 'models/'
dv_file = 'dv.bin'
model_file = 'model1.bin'
customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}

with open(os.path.join(dir, dv_file), 'rb') as f_in:
    dv = pickle.load(f_in)

with open(os.path.join(dir, model_file), 'rb') as f_in:
    model = pickle.load(f_in)

X = dv.transform([customer])
y_pred = model.predict_proba(X)[0, 1]
churn = y_pred >= 0.5

print(f'customer: {customer}')
print(f'churn_probability: {round(float(y_pred), 3)}, churn: {bool(churn)}')