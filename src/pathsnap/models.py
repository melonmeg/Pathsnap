"""Candidate-scoring models compared in the experiments (all expose predict_proba)."""
from collections import OrderedDict

from . import config as C

# name -> (algorithm family, human-readable key configuration, factory)
LGBM_DEFAULT = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=0.0)


def make_models(seed=C.SEED, n_jobs=-1):
    # imports are local so that a machine with a blocked sklearn.tree DLL can still use other models
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def _xgb():
        from xgboost import XGBClassifier
        return XGBClassifier

    def _trees():
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
        return ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier

    m = OrderedDict()
    m["LogisticRegression"] = ("Linear / probabilistic", "StandardScaler, L2 C=1.0, lbfgs, max_iter=300",
                               lambda: make_pipeline(StandardScaler(),
                                                     LogisticRegression(C=1.0, max_iter=300, random_state=seed)))
    m["GaussianNB"] = ("Linear / probabilistic", "var_smoothing=1e-9", lambda: GaussianNB())
    m["RandomForest"] = ("Ensemble (bagging)", "100 trees, max_depth=16, min_samples_leaf=10, max_samples=0.3",
                         lambda: _trees()[2](n_estimators=100, max_depth=16, min_samples_leaf=10,
                                                        max_samples=0.3, n_jobs=n_jobs, random_state=seed))
    m["ExtraTrees"] = ("Ensemble (randomised trees)", "100 trees, max_depth=16, min_samples_leaf=10, max_samples=0.3",
                       lambda: _trees()[0](n_estimators=100, max_depth=16, min_samples_leaf=10,
                                                    max_samples=0.3, bootstrap=True, n_jobs=n_jobs,
                                                    random_state=seed))
    m["HistGradientBoosting"] = ("Boosting (sklearn)", "lr=0.1, max_iter=200, max_leaf_nodes=31",
                                 lambda: _trees()[1](learning_rate=0.1, max_iter=200,
                                                                        max_leaf_nodes=31, random_state=seed))
    m["LightGBM"] = ("Boosting (GBDT)", "300 trees, lr=0.05, num_leaves=31, subsample=0.8, colsample=0.8",
                     lambda: LGBMClassifier(**LGBM_DEFAULT, random_state=seed, n_jobs=n_jobs, verbose=-1))
    m["XGBoost"] = ("Boosting (XGBoost)",
                    "300 trees, lr=0.05, max_depth=6, subsample=0.8, colsample=0.8, hist",
                    lambda: _xgb()(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
                                   colsample_bytree=0.8, tree_method="hist", eval_metric="logloss",
                                   n_jobs=n_jobs, random_state=seed))
    m["MLP"] = ("Neural network","StandardScaler, (64,32) ReLU, adam, alpha=1e-4, batch=1024, early stopping",
                lambda: make_pipeline(StandardScaler(),
                                      MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=1024, max_iter=40,
                                                    early_stopping=True, n_iter_no_change=5, random_state=seed)))
    return m
