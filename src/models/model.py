from sklearn import tree, linear_model, svm

from src.models.evaluation import evaluate


def test_models(X_train, X_test, y_train, y_test):
    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(X_train, y_train)

    evaluate(clf, X_test, y_test, 'Decision tree on normalised prepped data')

    reg = linear_model.LassoCV(cv=5)
    reg.fit(X_train, y_train)

    evaluate(reg, X_test, y_test, 'Lasso on normalised prepped data')

    regr = svm.SVR()
    regr.fit(X_train, y_train)

    evaluate(regr, X_test, y_test, 'SVM on normalised prepped data')
