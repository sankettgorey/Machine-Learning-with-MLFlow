import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error, mean_absolute_error

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient





def read_file(file):
    df = pd.read_csv(file)


    return df

def preprocessing(df):

    df["datetime"] = pd.to_datetime(df["datetime"])

    # Create new features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month

    # Create binary weather features
    df['is_clear_weather'] = (df['weather'] == 1).astype(int)
    df['is_rainy_weather'] = (df['weather'] >= 3).astype(int)

    df['is_holiday_workingday'] = ((df['holiday'] == 1) & (df['workingday'] == 1)).astype(int)

    df.drop(columns=["datetime"], inplace=True)

    return df


def split_data(df):
    X = df.drop(columns = ["count"], axis=1)
    y = df["count"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor(max_depth=10, random_state=42)

    with open("data.csv", "w") as f:
        for i in X_train.iterrows():
            f.write(i)

    # mlflow starts
    with mlflow.start_run():
        mlflow.log_param("model_type", "Decision_Tree_Regressor")
        mlflow.log_param("max_depth", 10)

        # train the model
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        feature_importance = model.feature_importances_

        plt.figure(figsize=(10, 6))
        plt.barh(X_train.columns, feature_importance)
        plt.title("Feature Importance")
        plt.savefig("feature_importance.png")

        mlflow.log_artifact(os.path.abspath("feature_importance.png"))

        # logging the model
        # mlflow.sklearn.log_model(model, "decision_tree_model")

        print("Model Logged Successfully")
        print('=' * 50)
        print(f"RMSE: {rmse}, MAE: {mae}")
        print('=' * 50)



def hyperparam_tuning(X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor(random_state=42)

    param_grid = {
        "max_depth": [5, 10 ,15],
        "min_samples_split": [2, 10, 20]
    }

    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid,
        cv=5, scoring="neg_mean_squared_error",
        verbose=1,
        n_jobs=-1
        )
    
    with mlflow.start_run(run_name="staging") as run:
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        mlflow.log_params(best_params)
        
        best_score = -grid_search.best_score_
        mlflow.log_metric("best_cross_val_score", best_score)

        test_preds = grid_search.best_estimator_.predict(X_test)

        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

        mlflow.log_metric("test_RMSE", test_rmse)


        mlflow.sklearn.log_model(
                grid_search.best_estimator_,
                artifact_path="model",
                registered_model_name="BikePredictionModel"
            )






def main():
    file = "bike-sharing-demand/train.csv"

    df = read_file(file)
    df = preprocessing(df)
    X_train, X_test, y_train, y_test = split_data(df)
    train_model(X_train, X_test, y_train, y_test)
    # hyperparam_tuning(X_train, X_test, y_train, y_test)



if __name__ == "__main__":
    main()