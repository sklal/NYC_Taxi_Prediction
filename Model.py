# Model

# Importing Libraries
# %%
from sklearn.linear_model import LinearRegression
from math import sqrt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


# Data Loading
# %%
df = pd.read_csv('D:\\NYC_Taxi_Prediction\\nyc_taxi_trip_duration.csv')

# Preprocessing & Feature Extraction

# Difference between pickup and dropoff latitude - will give an idea about the distance covered which could be predictive
# Difference between pickup and dropoff longitude - same reason as above
# Haversine distance between pickup and dropoff co-ordinates - to capture the actual distance travelled
# Pickup minute - since pickup hour is an important variable, the minute of pickup might well have been predictive
# Pickup day of year - same reason as above

# DateTime Conversion
# %%
# converting strings to datetime features
df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)
df['dropoff_datetime'] = pd.to_datetime(df.dropoff_datetime)

# Log transform the Y values
df_y = np.log1p(df['trip_duration'])

# Add some datetime features
df.loc[:, 'pickup_weekday'] = df['pickup_datetime'].dt.weekday
df.loc[:, 'pickup_hour_weekofyear'] = df['pickup_datetime'].dt.weekofyear
df.loc[:, 'pickup_hour'] = df['pickup_datetime'].dt.hour
df.loc[:, 'pickup_minute'] = df['pickup_datetime'].dt.minute
df.loc[:, 'pickup_dt'] = (df['pickup_datetime'] -
                          df['pickup_datetime'].min()).dt.total_seconds()
df.loc[:, 'pickup_week_hour'] = df['pickup_weekday'] * 24 + df['pickup_hour']

# Distance Features
# %%
# Eucledian Distance
# To get some idea on how far are the pickup and dropoff points

# displacement
y_dist = df['pickup_longitude'] - df['dropoff_longitude']
x_dist = df['pickup_latitude'] - df['dropoff_latitude']

# square distance
df['dist_sq'] = (y_dist ** 2) + (x_dist ** 2)

# distance
df['dist_sqrt'] = df['dist_sq'] ** 0.5

# Haversine Distance
# Let's calculate the distance (km) between pickup and dropoff points. The haversine formula determines the great-circle distance between two points on a sphere given their longitudes and latitudes.
# We will also calculate the approximate angle at which the dropoff location lies wrt the pickup location. pd.DataFrame.apply() would be too slow so the haversine function is rewritten to handle arrays.


def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def direction_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * \
        np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


df['haversine_distance'] = haversine_array(df['pickup_latitude'].values,
                                           df['pickup_longitude'].values,
                                           df['dropoff_latitude'].values,
                                           df['dropoff_longitude'].values)


df['direction'] = direction_array(df['pickup_latitude'].values,
                                  df['pickup_longitude'].values,
                                  df['dropoff_latitude'].values,
                                  df['dropoff_longitude'].values)


# Fastest route by road

# Here we will use data extracted from The Open Source Routing Machine or OSRM for each trip in our original dataset. OSRM is a C++ implementation of a high-performance routing engine for shortest paths in road networks. This will give us a very good estimate of distances between pickup and dropoff Pointsdatetime A combination of a date and a time. Attributes: ()
# %%
fr1 = pd.read_csv('D:\\NYC_Taxi_Prediction\\osrm\\fastest_routes_train_part_1\\fastest_routes_train_part_1.csv',
                  usecols=['id', 'total_distance', 'total_travel_time'])
fr2 = pd.read_csv('D:\\NYC_Taxi_Prediction\\osrm\\fastest_routes_train_part_2\\fastest_routes_train_part_2.csv',
                  usecols=['id', 'total_distance', 'total_travel_time'])

df_street_info = pd.concat((fr1, fr2))
df = df.merge(df_street_info, how='left', on='id')

df_street_info.head()

# Binning
# %%
# The lattitude and longitude could be a bit noisy and it might be a good idea to bin them and create new features after rounding their values.

### Binned Coordinates ###
df['pickup_latitude_round3'] = np.round(df['pickup_latitude'], 3)
df['pickup_longitude_round3'] = np.round(df['pickup_longitude'], 3)

df['dropoff_latitude_round3'] = np.round(df['dropoff_latitude'], 3)
df['dropoff_longitude_round3'] = np.round(df['dropoff_longitude'], 3)

# Other Features
# %%
# One Hot Encoding
# Here, Vendor ID can be converted to one hot encoding or frequency encoding since in the raw data it has values 1 and 2 without any inherent order.

df.vendor_id.value_counts()

# Now, there is not much difference in the frequencies of both and that might not make for an important feature. so we will just convert it to 0 and 1 by subtracting 1 from it

df['vendor_id'] = df['vendor_id'] - 1

np.sum(pd.isnull(df))

# For a route, the total distance and travel time are not available. Let's impute that with 0
df.fillna(0, inplace=True)


# Dropping the variables that should not be fed as features to the algorithms. We will drop

# id - Uniquely represents a sample in the train set
# pickup_datetime - Since we have extracted the datetime features, there is no need to keep the datetime column
# dropoff_datetime - If this is used to create features, it would be a leakage and we will get perfect model performance because The time gap between dropoff_datetime and pickup_datetime is essentially what we are trying to predict
# trip_duration - This is the target variable so needs to be dropped
# store_and_fwd_flag - This variable is not available before the start of the trip and should not be used for modelling.

# %%
df = df.drop(['id', 'pickup_datetime', 'dropoff_datetime',
              'trip_duration', 'store_and_fwd_flag'], axis=1)

# Model Building
# %%
df.head()


# Train-test split
# %%
# Splitting the data into Train and Validation set
xtrain, xtest, ytrain, ytest = train_test_split(
    df, df_y, test_size=1/3, random_state=0)

mean_pred = np.repeat(ytrain.mean(), len(ytest))

sqrt(mean_squared_error(ytest, mean_pred))

# Cross validation
# %%


def cv_score(ml_model, rstate=11, cols=df.columns):
    i = 1
    cv_scores = []
    df1 = df.copy()
    df1 = df[cols]

    kf = KFold(n_splits=5, random_state=rstate, shuffle=True)
    for train_index, test_index in kf.split(df1, df_y):
        print('\n{} of kfold {}'.format(i, kf.n_splits))
        xtr, xvl = df1.loc[train_index], df1.loc[test_index]
        ytr, yvl = df_y[train_index], df_y[test_index]

        model = ml_model
        model.fit(xtr, ytr)
        train_val = model.predict(xtr)
        pred_val = model.predict(xvl)
        rmse_score_train = sqrt(mean_squared_error(ytr, train_val))
        rmse_score = sqrt(mean_squared_error(yvl, pred_val))
        sufix = ""
        msg = ""
        #msg += "Train RMSE: {:.5f} ".format(rmse_score_train)
        msg += "Valid RMSE: {:.5f}".format(rmse_score)
        print("{}".format(msg))
        # Save scores
        cv_scores.append(rmse_score)
        i += 1
    return cv_scores


# Linear Regression
# %%
linreg_scores = cv_score(LinearRegression())

# Decision Trees
# %%
dtree_scores = cv_score(DecisionTreeRegressor(
    min_samples_leaf=25, min_samples_split=25))


# That's a lot of improvement. The reason for this could be the non linear relationship between the trip duration values and the location coordinates of pickup and dropoff points.


# Random Forest
# %%
rf_params = {'random_state': 0, 'n_estimators': 19,
             'max_depth': 11, 'n_jobs': -1, "min_samples_split": 43}
rf_scores = cv_score(RandomForestRegressor(**rf_params))


# XGBoost

# Looking at the performance of Random forest, it would be a good idea to try XGBoost which is based on gradient boosting techniques and check performance.
# First we will set the hyperparameters for XGBoost and use cross validation to track and figure out the correct number of rounds so that it does not overfit.
# Later, we will fit the XGBoost Regressor using the number of rounds identified from the above step and check the cross validation scores
# To find the approximate number of rounds for XGBoost, we will first create a validation set and check performance after each round.
# %%
dtrain = xgb.DMatrix(xtrain, label=ytrain)
dvalid = xgb.DMatrix(xtest, label=ytest)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

# %%
xgb_params = {}
xgb_params["objective"] = "reg:linear"
xgb_params['eval_metric'] = "rmse"
xgb_params["eta"] = 0.05
xgb_params["min_child_weight"] = 10
xgb_params["subsample"] = 0.9
xgb_params["colsample_bytree"] = 0.7
xgb_params["max_depth"] = 5
xgb_params['silent'] = 1
xgb_params["seed"] = 2019
xgb_params["nthread"] = -1
xgb_params["lambda"] = 2

xgb_model = xgb.train(xgb_params, dtrain, 10000, watchlist, early_stopping_rounds=50,
                      maximize=False, verbose_eval=20)
print('Modeling RMSE %.5f' % xgb_model.best_score)

# %%
xgb.plot_importance(xgb_model, max_num_features=28, height=0.7)

# So from top to bottom we see which features have the greatest effect on trip duration. It makes logical sense that the lattitude and longitude have high impact on predicting the trip duration. The further you travel, the longer it'll take. Interestingly, day of month is ranked higher here than day of week.

xgb_params['num_round'] = xgb_model.best_iteration
xgb_model_final = xgb.XGBRegressor()
xgb_scores = cv_score(xgb_model_final)

# %%
results_df = pd.DataFrame({'linear_regression': linreg_scores,
                           'dtree': dtree_scores, 'RF': rf_scores, 'XGB': xgb_scores})

results_df.plot(y=["linear_regression", "dtree", 'RF',
                   'XGB'], kind="bar", legend=False)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
