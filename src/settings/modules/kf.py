from sklearn.model_selection import RepeatedKFold

def get_kfold_class(split_count: int, loop_count: int, seed: int) -> RepeatedKFold:
    kfold = RepeatedKFold(
        n_splits=split_count,
        n_repeats=loop_count,
        random_state=seed
    )
    return kfold
