# data
import nfl_data_py as nfl

# data loading and plotting
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import load_data, prepare_features

# models
import xgboost as xgb
from xgboost import XGBRegressor, plot_importance
from sklearn.linear_model import LinearRegression

# interpretation
import shap
from interpret import show
from scipy.stats import spearmanr

# pipeline
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import RFECV, RFE
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, nan_euclidean_distances


def split_data(df_merged):

    X_train = df_merged[df_merged['season'].isin(list(range(2010, 2024)))].drop(columns=['fantasy_points'])
    y_train = df_merged[df_merged['season'].isin(list(range(2010, 2024)))]['fantasy_points']

    X_val = df_merged[df_merged['season']==2023].drop(columns=['fantasy_points'])
    y_val = df_merged[df_merged['season']==2023]['fantasy_points']

    X_test = df_merged[df_merged['season']==2024].drop(columns=['fantasy_points'])
    y_test = df_merged[df_merged['season']==2024]['fantasy_points']

    return X_train, X_test, y_train, y_test



def create_preprocessor(df_merged):

    X = df_merged.drop(columns=['fantasy_points'])

    numeric_features = X.select_dtypes(include=['int32', 'int64', 'float32', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor



def create_pipeline(model, preprocessor):

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline



def optimize_hyperparameters(model, param_grid, X_train, y_train, preprocessor):

    pipeline = create_pipeline(model, preprocessor)
    
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=TimeSeriesSplit(n_splits=3), verbose=2)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_}")
    
    return grid_search.best_estimator_


def evaluate_models(model, X_test, y_test):
    metrics = {
        'mean_absolute_error': mean_absolute_error,
        'mean_squared_error': mean_squared_error,
        'root_mean_squared_error': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2_score': r2_score,
        'spearman_rank_correlation': lambda y_true, y_pred: spearmanr(y_true, y_pred)[0]
    }

    y_pred = model.predict(X_test)
    model_name = model.named_steps['model'].__class__.__name__
    print(f"Evaluation results for model: {model_name}")
    for metric_name, metric_func in metrics.items():
        score = metric_func(y_test, y_pred)
        print(f"{metric_name}: {score}")
    print("\n")



import shap
from scipy.stats import spearmanr

def plot_feature_importances(model, X_train):
    model_name = model.named_steps['model'].__class__.__name__
    print(f"Plotting feature importances for model: {model_name}")

    # Plot feature importances using SHAP
    explainer = shap.Explainer(model.named_steps['model'], X_train)
    shap_values = explainer(X_train)

    if model_name == 'XGBRegressor':
        # Hole die Feature-Wichtigkeiten
        feature_importances = model.named_steps['model'].feature_importances_
        # Erstelle ein DataFrame für Features und deren Wichtigkeiten
        feature_importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': feature_importances
        })
        # Sortiere die Feature-Wichtigkeiten absteigend
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Wähle die Top 20 Features
        top_features = feature_importance_df.head(20)

        # Plotten der Feature-Wichtigkeiten
        plt.figure(figsize=(10, 8))
        plt.barh(top_features['Feature'], top_features['Importance'])
        plt.gca().invert_yaxis()  # Um die höchste Wichtigkeit oben anzuzeigen
        plt.xlabel("Feature Importance")
        plt.ylabel("Features")
        plt.title(f"Top 20 Feature Importances in {model_name}")
        plt.show()

    # Summary plot
    shap.summary_plot(shap_values, X_train)

    # Bar plot
    shap.summary_plot(shap_values, X_train, plot_type="bar")

    print("\n")



import joblib

def save_model(model, model_name):
    # Speicherpfad definieren
    file_path = f"models/{model_name}_model.pkl"
    joblib.dump(model, file_path)
    print(f"Modell '{model_name}' wurde unter '{file_path}' gespeichert.")



def load_model(model_name):
    # Speicherpfad definieren
    file_path = f"models/{model_name}_model.pkl"
    model = joblib.load(file_path)
    print(f"Modell '{model_name}' wurde aus '{file_path}' geladen.")
    return model



def predict_and_merge(model_name, df_merged):
    # Load the model
    model = load_model(model_name)
    
    # Filter the data for the 2024 season
    df_test = df_merged[df_merged['season'] == 2024]
    
    # Prepare the data for prediction
    X_test = df_test.drop(columns=['fantasy_points'])
    y_test = df_test['fantasy_points']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Create the output dataframe
    df_output = df_test[['season', 'week', 'player_id', 'position', 'recent_team', 'opponent_team']].copy()
    df_output['predicted_fantasy_points'] = y_pred
    df_output['actual_fantasy_points'] = y_test

    return df_output



def main():
    # Prepare the data
    df = load_data()
    df_merged = prepare_features(df)
    # df_output, df_merged = prepare_features()
    X_train, X_test, y_train, y_test = split_data(df_merged)
    preprocessor = create_preprocessor(df_merged)

    # Define the models and their hyperparameters
    models = {
        'XGBoost': (XGBRegressor(), {
            'model__n_estimators': [100, 500],
            'model__max_depth': [3, 6],
            'model__learning_rate': [0.05, 0.1]
        }),
        'Linear Regression': (LinearRegression(), {
            'model__fit_intercept': [True, False],
            'model__n_jobs': [1, -1]
        })
    }

    # Run the pipeline for each model
    for model_name, (model, param_grid) in models.items():
        print(f"Performing Grid Search for {model_name}...")
        best_model = optimize_hyperparameters(model, param_grid, X_train, y_train, preprocessor)
        print(f"Evaluating {model_name}...")
        evaluate_models(best_model, X_test, y_test)
        plot_feature_importances(best_model, X_train)
        save_model(best_model, model_name)

if __name__ == "__main__":
    main()