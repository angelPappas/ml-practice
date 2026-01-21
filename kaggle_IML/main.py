from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess

# from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
# from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.experiments.logger import log_experiment
from sklearn.metrics import mean_absolute_error

# Load Raw Data
melbourne_data = load_raw_data()

# Process Data
X, y = preprocess(melbourne_data)

# Select Model
model = DecisionTreeRegressor()

# Validate Model
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
model.fit(train_X, train_y)

# Fianlly train model again with all data
mae = mean_absolute_error(model.predict(val_X), val_y)

log_experiment(
    features=X.columns,
    model=type(model),
    hyperparameters=model.get_params(),
    metric="mae",
    metric_value=mae)
