import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Protocol, Dict
from Models.ANN import ANN
from Models.CNN import CNN
from Models.LightGBM import LightGBM
from Models.lstm import lstm
from Models.Xgboost import Xgboost
import warnings
warnings.filterwarnings('ignore')

class DataSplitter:
    def __init__(self, test_size=0.2, val_size=0.2, random_state=42):
        """
        Initialize the DataSplitter.

        Parameters:
            test_size : float, optional (default=0.2)
                The proportion of the dataset to include in the test split.

            val_size : float, optional (default=0.2)
                The proportion of the dataset to include in the validation split.

            random_state : int or RandomState, optional (default=42)
                Seed for the random number generator for reproducible results.
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split_data(self, df=None, y_col=None, X=None, y=None, shuffle=False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split the data into training and testing sets.

        Parameters:
            X : pandas DataFrame or array-like, optional
                The feature matrix.

            y : pandas Series or array-like, optional
                The target vector.

            df : pandas DataFrame, optional
                The entire DataFrame containing both features and target.

            y_col : str, optional
                The column name of the target variable in the DataFrame.

            shuffle : bool, optional

        Returns:
            Tuple of arrays: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if df is not None and y_col is not None:
            X = df.drop(columns=[y_col])
            y = df[y_col]
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, shuffle=shuffle)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=self.test_size, random_state=self.random_state, shuffle=shuffle)

        return X_train, X_val, X_test, y_train, y_val, y_test



class ModelTrainer:
    def __init__(self, model, X_train, y_train, X_val, y_val, X_test, y_test, params=None):
        """
        Initialize the ModelTrainer.

        Parameters:
            model : str
                The type of model to be trained.

            X_train : pandas DataFrame
                The training feature matrix.

            y_train : pandas Series
                The training target vector.

            X_val : pandas DataFrame
                The validation feature matrix.

            y_val : pandas Series
                The validation target vector.

            X_test : pandas DataFrame
                The testing feature matrix.

            y_test : pandas Series
                The testing target vector.

            params : dict, optional
                Hyperparameters for the model.
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.params = params
        self.trained_model = None

    def train_model(self) -> None:
        """
        Train the specified model.

        If hyperparameters are provided, train the model with those parameters.
        Otherwise, perform hyperparameter tuning and train the model with the best parameters.
        """
        model_obj = None
        if self.model == 'ANN':
            model_obj = Xgboost(self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test)
        elif self.model == 'CNN':
            model_obj = CNN(self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test)
        elif self.model == 'LightGBM':
            model_obj = LightGBM(self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test)
        elif self.model == 'lstm':
            model_obj = lstm(self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test)
        elif self.model == 'Xgboost':
            model_obj = Xgboost(self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test)

        if model_obj is not None:
            if self.params:
                self.trained_model = model_obj.train(self.params, self.X_train, self.y_train, self.X_val, self.y_val)
            else:
                best_params = model_obj.tune(n_trials=50)
                self.trained_model = model_obj.train(best_params, self.X_train, self.y_train, self.X_val, self.y_val)
        else:
            raise ValueError("Invalid model type. Please provide a valid model type.")

    def predict_model(self, test_data=None) -> pd.Series:
        """
        Predict using the trained model.

        Parameters:
            test_data : pandas DataFrame, optional
                The feature matrix for prediction. If not provided, uses the test set used for training.

        Returns:
            pandas Series
                Predictions from the model.
        """
        if test_data is None:
            if self.X_test is None:
                raise ValueError("No test data provided.")
            test_data = self.X_test
        if self.trained_model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        preds = self.trained_model.predict(test_data)

        return preds

if __name__ == "__main__":
    # Example usage
    model = 'Xgboost'  # Change this to the desired model
    model_trainer = ModelTrainer(model, X_train, y_train, X_val, y_val, X_test, y_test)
    model_trainer.train_model()
    predictions = model_trainer.predict_model(new_test_data)
    print(predictions)


