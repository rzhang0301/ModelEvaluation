import numpy as np
import pandas as pd
from model_training import ModelTrainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
from abc import ABC, abstractmethod
from typing import Protocol, Dict
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator(ABC):
    """
    Abstract base class for model evaluation. 
    """

    def __init__(self, models):
        self.models = models
        self.classification_results = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score'])
        self.regression_results = pd.DataFrame(columns=['Model', 'Mean Squared Error', 'Mean Absolute Error', 'R^2 Score', 'Root Mean Squared Error'])

    @abstractmethod
    def evaluate(self, X_test, y_test) -> None:
        """
        Abstract method for evaluating models.

        Parameters:
            X_test : pandas DataFrame
                The feature matrix of the dataset.

            y_test : pandas Series
                The target vector of the dataset.

        Returns:
            None
        """
        pass

    def output_results(self, result_type) -> pd.DataFrame:
        """
        Output the evaluation results.

        Parameters:
            result_type : str
                Either 'classification' or 'regression' to specify the type of results to output.

        Returns:
            pd.DataFrame
                The evaluation results data frame.
        """
        if result_type == 'classification':
            return self.classification_results
        elif result_type == 'regression':
            return self.regression_results
        else:
            raise ValueError("Invalid result_type. Use 'classification' or 'regression'.")



class ClassificationModelEvaluator(ModelEvaluator):
    def evaluate(self, X_test, y_test) -> None:
        """
        Evaluate classification models based on the specific evaluator type.

        Parameters and logic are inherited from ModelEvaluator.
        """

        for name, model in self.models.items():
            # model.fit(X_train, y_train)
            # y_pred = model.predict(X_test)
            y_pred = model.predict_model(X_test)

            # Evaluation metrics for classification
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            roc_auc = roc_auc_score(y_test, y_pred)

            model_results = pd.DataFrame({'Model': [name], 'Accuracy': [accuracy],
                                            'Precision': [precision], 'Recall': [recall],
                                            'F1 Score': [f1], 'ROC AUC Score': [roc_auc]})

            self.classification_results = pd.concat([self.classification_results, model_result], ignore_index=True)


class RegressionModelEvaluator(ModelEvaluator):
    def evaluate(self, X_test, y_test) -> None:
        """
        Evaluate regression models based on the specific evaluator type.

        Parameters and logic are inherited from ModelEvaluator.
        """

        for name, model in self.models.items():
            # model.fit(X_train, y_train)
            # y_pred = model.predict(X_test)
            y_pred = model.predict_model(X_test)

            # Evaluation metrics for regression
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)

            model_results = pd.DataFrame({'Model': [name], 'Mean Squared Error': [mse],
                                          'Mean Absolute Error': [mae], 'R^2 Score': [r2],
                                          'Root Mean Squared Error': [rmse]})
            self.regression_results = pd.concat([self.regression_results, model_results], ignore_index=True)

            
    def plot_residuals(self, model, X_test, y_test, title):
        """
        Plot residuals for regression models using Plotly.

        Parameters:
            model : object
                The trained regression model.

            X_test : pandas DataFrame
                The feature matrix of the test set.

            y_test : pandas Series
                The true target values of the test set.

            title : str
                The title of the plot.

        Returns:
            None
        """
        residuals = y_test - model.predict(X_test)

        # Create a scatter plot using Plotly
        fig = px.scatter(x=model.predict(X_test), y=residuals, title=title,
                         labels={'x': 'Predicted Values', 'y': 'Residuals'})
        fig.add_shape(type='line', line=dict(color='red', dash='dash'),
                      x0=min(model.predict(X_test)), x1=max(model.predict(X_test)),
                      y0=0, y1=0)
        fig.show()

