RANDOM_STATE = 42


CATBOOST_PARAMS = {
    'task_type': 'CPU',
    'loss_function': 'Logloss',
    'eval_metric': 'F1',
    'custom_metric': ['F1', 'Precision', 'Recall'],
    'random_seed': 42,
    'early_stopping_rounds': 100,
    'od_type': 'IncToDec'
}

