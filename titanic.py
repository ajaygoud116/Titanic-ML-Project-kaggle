import pandas as pd
import numpy as np
# Load datasets
train = pd.read_csv("train.csv")
x_test = pd.read_csv("test.csv")


train.replace(["male"], [1], inplace=True)
train.replace(["female"], [0], inplace=True)
x_test.replace(["male"], [1], inplace=True)
x_test.replace(["female"], [0], inplace=True)


y_train = train["Survived"]


train.drop(columns=['Name', 'Survived', "Ticket", "Embarked", "Cabin"], inplace=True)
x_test.drop(columns=['Name', "Ticket", "Embarked", "Cabin"], inplace=True)


train["Age"] = train.groupby("Pclass")["Age"].transform(lambda x: x.fillna(x.median()))
x_test["Age"] = x_test.groupby("Pclass")["Age"].transform(lambda x: x.fillna(x.median()))


x_test["Fare"].fillna(x_test["Fare"].median(), inplace=True)


X_train = train.to_numpy()
y_train = y_train.to_numpy().reshape(-1, 1)
X_test = x_test.to_numpy()


w = np.zeros((X_train.shape[1], 1))
b = 0


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_model_output(X, w, b):
    return sigmoid(np.dot(X, w) + b)


def compute_cost(X, y, w, b):
    m = X.shape[0]
    f_wb = compute_model_output(X, w, b)
    cost = -(1/m) * np.sum(y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb))
    return cost


def compute_gradient(X, y, w, b):
    m = X.shape[0]
    f_wb = compute_model_output(X, w, b)
    dw = (1/m) * np.dot(X.T, (f_wb - y))
    db = (1/m) * np.sum(f_wb - y)
    return dw, db


def gradient_descent(X, y, w, b, alpha, num_iters):
    for i in range(num_iters):
        dw, db = compute_gradient(X, y, w, b)
        w -= alpha * dw
        b -= alpha * db
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {compute_cost(X, y, w, b)}")
    return w, b


alpha = 0.01  
num_iters = 2000 
w, b = gradient_descent(X_train, y_train, w, b, alpha, num_iters)


y_pred = compute_model_output(X_test, w, b)
y_pred = (y_pred >= 0.5).astype(int)  


submission = pd.DataFrame({"PassengerId": x_test["PassengerId"], "Survived": y_pred.flatten()})
submission.to_csv("submission.csv", index=False)

print("Submission file saved as 'submission.csv'.")
