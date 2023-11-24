import mlflow

f1_score_filter = "metrics.score > 0.010"

result = mlflow.search_runs(
    experiment_names=["LR experiment"],
    filter_string=f1_score_filter,
    order_by=["metrics.score DESC"]
)

print(result)