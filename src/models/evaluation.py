from sklearn import linear_model
from sklearn.model_selection import cross_validate


def get_baseline_model(X_trian, y_train):
    reg = linear_model.LinearRegression()
    reg.fit(X_trian, y_train)
    return reg


def evaluate(model, X_test, y_test, name):
    scores = cross_validate(model, X_test, y_test, cv=3, scoring=('r2', 'neg_mean_squared_error'),
                            return_train_score=True)

    print('Model {} got a r2 score of {} and negative mean squared error {}'.format(name, scores['train_r2'],
                                                                                    scores[
                                                                                        'test_neg_mean_squared_error']))
