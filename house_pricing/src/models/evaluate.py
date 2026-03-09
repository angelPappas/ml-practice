from sklearn.metrics import mean_absolute_error


def evaluate_model(model, val_X, val_y):
    return mean_absolute_error(val_y, mean_absolute_error(val_X))
