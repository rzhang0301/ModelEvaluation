from model_evaluations import RegressionModelEvaluator
from model_training import DataSplitter, ModelTrainer
import pandas as pd

X_regression = pd.read_pickle('./X.pkl')
y_regression = pd.read_pickle('./y.pkl')
splitter = DataSplitter(test_size=0.2, random_state=42)
X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_data(X=X_regression, y=y_regression)

models = ['Xgboost']
model_trainers = {}
for model in models:
    model_trainer = ModelTrainer(model, X_train, y_train, X_val, y_val, X_test, y_test)
    model_trainer.train_model()
    model_trainers[model] = model_trainer

regression_evaluator = RegressionModelEvaluator(model_trainers, X_test, y_test)
regression_evaluator.evaluate(X_test, y_test)
regression_results = regression_evaluator.output_results('regression')
print(regression_results)