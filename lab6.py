from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from plot_diagram import plot_projected_and_expected
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


def stacking(x_train, x_test, y_train, y_test):
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
        ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42, dual=False)))
    ]
    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    clf.fit(x_train, y_train)
    plot_projected_and_expected(x_test, y_test, clf.predict(x_test))
    print("stacking точность классификации= ", clf.score(x_test, y_test))


def random_forest(x_test, y_test, x_train, y_train, num_trees, max_features, kfold):
    model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
    results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold)
    model.fit(x_train, y_train)
    plot_projected_and_expected(x_test, y_test, model.predict(x_test))
    print("Random Forest точность классификации= ", results.mean())


def bagging(x_train, x_test, y_train, y_test):
    clf = BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0)
    clf.fit(x_train, y_train)
    plot_projected_and_expected(x_test, y_test, clf.predict(x_test))
    print("bagging точность классификации= ", clf.score(x_test, y_test))


def adaboost(x_train, x_test, y_train, y_test, num_trees, seed):
    clf = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
    clf.fit(x_train, y_train)
    plot_projected_and_expected(x_test, y_test, clf.predict(x_test))
    print("AdaBoost точность классификации= ", clf.score(x_test, y_test))


def xgboost(x_train, x_test, y_train, y_test):
    clf = XGBClassifier(use_label_encoder=False)
    clf.fit(x_train, y_train)
    plot_projected_and_expected(x_test, y_test, clf.predict(x_test))
    print("XGBoost точность классификации= ", clf.score(x_test, y_test))


def catboost(x_train, x_test, y_train, y_test, num_trees):
    clf = CatBoostClassifier(verbose=0, n_estimators=num_trees)
    clf.fit(x_train, y_train)
    plot_projected_and_expected(x_test, y_test, clf.predict(x_test))
    print("CatBoost точность классификации= ", clf.score(x_test, y_test))

def lightgbm(x_train, x_test, y_train, y_test):
    clf = LGBMClassifier()
    clf.fit(x_train, y_train)
    plot_projected_and_expected(x_test, y_test, clf.predict(x_test))
    print("LightGBM точность классификации= ", clf.score(x_test, y_test))

def stochastic_gradient_boosting(x_test, y_test, x_train, y_train, num_trees, seed, kfold):
    model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
    results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold)
    model.fit(x_train, y_train)
    plot_projected_and_expected(x_test, y_test, model.predict(x_test))
    print("Средняя оценка точности классификации (Stochastic Gradient Boosting)= ", results.mean())


def lab6(x_train, x_test, y_train, y_test):
    seed = 7
    num_trees = 100  # колличество деревьев
    max_features = 7  # 7 объектов
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)

    print('Bagging')
    bagging(x_train, x_test, y_train, y_test)
    random_forest(x_test, y_test, x_train, y_train, num_trees, max_features, kfold)

    print('Boosting')
    adaboost(x_train, x_test, y_train, y_test, num_trees, seed)
    xgboost(x_train, x_test, y_train, y_test)
    catboost(x_train, x_test, y_train, y_test, num_trees)
    lightgbm(x_train, x_test, y_train, y_test)

    print('Stacking')
    stacking(x_train, x_test, y_train, y_test)
