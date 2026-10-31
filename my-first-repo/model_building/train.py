import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn
import os

# Set tracking server paths
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Tourism_Package_Prediction")

# Load data files
Xtrain = pd.read_csv("Xtrain.csv")
ytrain = pd.read_csv("ytrain.csv").iloc[:, 0]
Xtest = pd.read_csv("Xtest.csv")
ytest = pd.read_csv("ytest.csv").iloc[:, 0]

# Feature Groups
numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar", 
    "NumberOfChildrenVisiting", "MonthlyIncome"
]

categorical_features = [
    "TypeofContact", "Occupation", "Gender", 
    "ProductPitched", "MaritalStatus", "Designation"
]

# Balance weights
class_weight = (len(ytrain) - sum(ytrain)) / sum(ytrain)

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

xgb_model = XGBClassifier(scale_pos_weight=class_weight, random_state=42, use_label_encoder=False, eval_metric="logloss")
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Explicitly defined search space grid 
param_grid = {}
param_grid["xgbclassifier__n_estimators"] = 
param_grid["xgbclassifier__max_depth"] = 
param_grid["xgbclassifier__colsample_bytree"] = [0.8, 1.0]
param_grid["xgbclassifier__subsample"] = [0.8, 1.0]
param_grid["xgbclassifier__learning_rate"] = [0.05, 0.1]
param_grid["xgbclassifier__reg_lambda"] = [1.0, 5.0]

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring="f1")
    grid_search.fit(Xtrain, ytrain)

    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            params = {k.replace("xgbclassifier__", ""): v for k, v in results["params"][i].items()}
            mlflow.log_params(params)
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])

    best_params = {k.replace("xgbclassifier__", ""): v for k, v in grid_search.best_params_.items()}
    mlflow.log_params(best_params)
    best_model = grid_search.best_estimator_

    classification_threshold = 0.5
    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= classification_threshold).astype(int)
    y_pred_test = (best_model.predict_proba(Xtest)[:, 1] >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1_score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1_score": test_report["1"]["f1-score"]
    })

    # Save the deployment artifact inside your production-facing repository structural folders
    model_path = "my-first-repo/model"
    mlflow.sklearn.log_model(best_model, "model")
    print("Production training file written successfully.")
