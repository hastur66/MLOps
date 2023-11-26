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

lr = LinearRegression(n_jobs=1)

lr.fit(X_train, y_train)

# save to local file-system 
mlflow.sklearn.save_model(lr, "./mlruns/models")

# log as a artifact
mlflow.sklearn.log_model(lr, "./mlruns/models")

# load model
mlflow.sklearn.load_model(model_uri="./mlruns/models")