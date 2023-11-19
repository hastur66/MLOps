import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

mlflow.set_experiment("LR experiment")

mlflow.start_run()

data = load_diabetes()

X = data.data

y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

lr = LogisticRegression(n_jobs=1)

lr.fit(X_train, y_train)

score = lr.score(X, y)

mlflow.log_metric("score", score)

mlflow.log_param("n_jobs", 1)

mlflow.log_artifact("logging_run.py")