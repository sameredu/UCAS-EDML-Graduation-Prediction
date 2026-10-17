"""The three classifiers, built from the hyperparameters in config.py."""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from . import config as C

_BUILDERS = {
    "Random Forest": lambda p, s: RandomForestClassifier(random_state=s, n_jobs=-1, **p),
    "Decision Tree": lambda p, s: DecisionTreeClassifier(random_state=s, **p),
    "GBT Classifier": lambda p, s: GradientBoostingClassifier(random_state=s, **p),
}


def build(name: str, seed: int = C.SEED):
    return _BUILDERS[name](C.HYPERPARAMETERS[name], seed)


def build_all(seed: int = C.SEED) -> dict:
    return {name: build(name, seed) for name in C.MODEL_ORDER}
