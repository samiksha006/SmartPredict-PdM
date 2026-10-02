from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, VotingClassifier, StackingClassifier
)
from xgboost import XGBClassifier

def get_models(random_state=42):
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state, class_weight="balanced"
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Linear SVM": SVC(
            kernel="linear", probability=True, class_weight="balanced", random_state=random_state
        ),
        "Polynomial SVM": SVC(
            kernel="poly", degree=3, C=1.0, gamma="scale",
            probability=True, class_weight="balanced", random_state=random_state
        ),
        "RBF SVM": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, class_weight="balanced", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, class_weight="balanced",
            n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, random_state=random_state, class_weight="balanced",
            n_jobs=-1
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200, learning_rate=0.05, random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=random_state,
            n_jobs=-1
        ),
    }
