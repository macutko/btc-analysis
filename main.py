from sklearn.model_selection import train_test_split

from src.correlation.correlation import get_heat_map
from src.data_prep import get_clean_data
from src.models.evaluation import get_baseline_model, evaluate
from src.models.model import test_models

if __name__ == '__main__':
    data = get_clean_data()
    X_train, X_test, y_train, y_test = train_test_split(data.drop(['BTC Price', 'date'], axis=1), data['BTC Price'],
                                                        shuffle=True,
                                                        test_size=0.4, random_state=42)

    baseline_model = get_baseline_model(X_train, y_train)
    evaluate(baseline_model, X_test, y_test, 'baseline model')

    # Based on this heat map we conclude that the Number of TX is the top KING of Pop

    get_heat_map(
        X_train.drop(['Gold in USD', 'Ethereum Price', 'Litecoin Price', 'Nasdaq composite index', 'DJI'], axis=1),
        y_train, 'Tech Features')

    # This heat map gives us Gold, LTC, DJI
    get_heat_map(
        X_train.drop(
            ['BTC network hashrate', 'Average BTC block size', 'NUAU - BTC', 'Number TX - BTC', 'Difficulty - BTC',
             'TX fees - BTC', 'Estimated TX Volume USD - BTC'], axis=1),
        y_train, 'Asset classes')

    chosen_features = ['Gold in USD', 'Litecoin Price', 'DJI', 'Number TX - BTC']

    # X_train = normalise_data(X_train)
    # X_test = normalise_data(X_test)

    # # get a simple linear Reg model for baseline testing with normalised data
    # baseline_model_norm = get_baseline_model(X_train, y_train)
    # evaluate(baseline_model_norm, X_test, y_test, 'baseline model normalised')
    #
    # X_train = X_train[chosen_features]
    # X_test = X_test[chosen_features]

    test_models(X_train, X_test, y_train, y_test)
