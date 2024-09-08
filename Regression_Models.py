# ----------- import Necessary packages -----

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split


def Preprocessing():
    # load the file
    data = pd.read_csv('2016.csv')

    # ---- replace Nan values
    data.fillna(0, inplace=True)
    print(data.columns)
    print(data.shape)

    # mean of the arrival delay
    avg_delay = data['ARR_DELAY'].mean()

    # create new dataframe by evaluate arrival delay for each carrier in OP_CARRIER column
    airline_delay_df = pd.DataFrame({'avg_delay': data.groupby(["OP_CARRIER"])['ARR_DELAY'].mean()}).reset_index()
    n_data = pd.merge(data, airline_delay_df, how='inner', on='OP_CARRIER')

    # Converting flight date to a Datetime object and then computing which weekday the flight was on
    n_data['FL_DATE'] = pd.to_datetime(n_data['FL_DATE'])
    # n_data['flight_weekday'] = n_data['FL_DATE'].apply(lambda x: x.weekday() + 1)
    n_data['FL_weekday'] = n_data['FL_DATE'].apply(lambda x: x.weekday())

    # categorical column to numeric columns
    n_data = pd.get_dummies(n_data, columns=['OP_CARRIER'])
    n_data = n_data.drop(['FL_DATE', 'ORIGIN', 'DEST', 'CANCELLATION_CODE'], axis=1)
    n_data.to_csv('processed.csv')  # save to csv file

    # ----------------------------------- correlation coefficient of the models ---------------
    corr = n_data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    # Save
    plt.savefig('heatmap.png', dpi=500, bbox_inches='tight')

    # Show the plot (optional)
    plt.show()


class Prediction_Models:
    def __init__(self, xtrain, xtest, ytrain, ytest):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest

    def Evaluation_Metrics(self, y_true, predicted):
        """
        :param y_true: array - true values from data
        :param predicted: array - predicted values from the trained regression model
        :return: list of metrics (MAE, MSE, RMSE, R2)
        """
        # Mean Absolute Error
        MAE = np.mean(abs(y_true - predicted))
        # Mean Squared Error
        MSE = mean_squared_error(y_true, predicted)
        # Root Mean Squared Error
        RMSE = np.sqrt(MSE)
        # coefficient of determination
        R2 = r2_score(y_true, predicted)
        return [MAE, MSE, RMSE, R2]

    def print_results(self, x, regressor):

        print(f'MAE  for {regressor}------ > {x[0]}')
        print(f'MSE  for {regressor}------ > {x[1]}')
        print(f'RMSE  for {regressor}------ > {x[2]}')
        print(f'R2  for {regressor}------ > {x[3]}')
        print('---------------------------------------------------------------------------')

    def linear_regression(self):
        model = LinearRegression()
        # training
        model.fit(self.xtrain, self.ytrain)
        # predict
        preds = model.predict(self.xtest)
        metrics = self.Evaluation_Metrics(self.ytest, preds)
        self.print_results(metrics, 'Linear Regression')

    def svr(self):
        model = SVR()
        # Train the model
        model.fit(self.xtrain, self.ytrain)
        # Predict
        preds = model.predict(self.xtest)
        metrics = self.Evaluation_Metrics(self.ytest, preds)
        self.print_results(metrics, 'Support Vector Regression')

    def xgboost_regression(self):
        model = XGBRegressor()
        # Train the model
        model.fit(self.xtrain, self.ytrain)
        # Predict
        preds = model.predict(self.xtest)
        metrics = self.Evaluation_Metrics(self.ytest, preds)
        self.print_results(metrics, 'eXtreme Gradient Boosting Regression')

    def decision_tree_regression(self):
        model = DecisionTreeRegressor()
        # Train the model
        model.fit(self.xtrain, self.ytrain)
        # Predict
        preds = model.predict(self.xtest)
        metrics = self.Evaluation_Metrics(self.ytest, preds)
        self.print_results(metrics, 'Decision Tree Regression')

    def random_forest_regression(self):
        model = RandomForestRegressor()
        # Train the model
        model.fit(self.xtrain, self.ytrain)
        # Predict
        preds = model.predict(self.xtest)
        metrics = self.Evaluation_Metrics(self.ytest, preds)
        self.print_results(metrics, 'Random Forest Regression')


if __name__ == "__main__":
    # preprocess the dataframe
    Preprocessing()
    data = pd.read_csv('processed.csv')
    labels = data['ARR_DELAY']
    features = data.drop(['ARR_DELAY'], axis=1)
    print(features.shape)
    print(labels.shape)
    
    # split the data
    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.2, random_state=42)

    models = Prediction_Models(xtrain, xtest, ytrain, ytest)

    # regression models
    models.linear_regression()
    models.svr()
    models.xgboost_regression()
    models.decision_tree_regression()
    models.random_forest_regression()
