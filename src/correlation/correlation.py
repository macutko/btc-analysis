import matplotlib.pyplot as plt


def get_heat_map(X_train, y_train, name):
    y_train.reset_index(drop=True, inplace=True)
    X_train['BTC Price'] = y_train

    f = plt.figure(figsize=(19, 15))
    plt.matshow(X_train.corr(), fignum=f.number)
    plt.xticks(range(X_train.shape[1]), X_train.columns, fontsize=14, rotation=45)
    plt.yticks(range(X_train.shape[1]), X_train.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix {}'.format(name), fontsize=16)
    # plt.show()
