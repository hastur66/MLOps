import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://localhost:8080")

mlflow.sklearn.autolog()


data = load_diabetes()

X = data.data

y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

lr = LinearRegression()

lr.fit(X_train, y_train)