from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from pipeline.O1_data_preparation import load_data, prepare_features, split_data
from pipeline.O2_pipeline_implementation import create_preprocessor, optimize_hyperparameters, save_model
from pipeline.O3_model_evaluation import evaluate_models, plot_feature_importances


def main():

    # Prepare features for model training and testing
    df = load_data()
    df_merged = prepare_features(df)

    # Split data for training and testing
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

    # Run pipeline for each model
    for model_name, (model, param_grid) in models.items():
        print(f"Performing Grid Search for {model_name}...")
        best_model = optimize_hyperparameters(model, param_grid, X_train, y_train, preprocessor)
        print(f"Evaluating {model_name}...")
        evaluate_models(best_model, X_test, y_test)
        plot_feature_importances(best_model, X_train)
        save_model(best_model, model_name)



if __name__ == "__main__":
    main()