RANDOM_STATE = 42


CATBOOST_PARAMS = {
    "task_type": "CPU",
    "loss_function": "Logloss",
    "eval_metric": "F1",
    "custom_metric": ["F1", "Precision", "Recall"],
    "random_seed": 42,
    "od_type": "IncToDec",
}


CB_PARAM_GRID = {
    "learning_rate": [0.2, 0.3],
    "depth": [5, 6, 7],
    "l2_leaf_reg": [8, 10],
    "iterations": [100, 200, 300, 400],
}
