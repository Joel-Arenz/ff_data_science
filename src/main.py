from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from pipeline.O1_data_preparation import load_ind_data, prepare_ind_features, load_sys_data, prepare_sys_features, split_data
from pipeline.O2_pipeline_implementation import create_preprocessor, optimize_hyperparameters, save_model
from pipeline.O3_model_evaluation import evaluate_models, plot_feature_importances


def main():
    # Prepare individual features for model training and testing
    df_ind = load_ind_data()
    df_ind_prepared = prepare_ind_features(df_ind)

    # Prepare systematic features for model training and testing
    df_sys = load_sys_data()
    df_sys_prepared = prepare_sys_features(df_sys)

    # Split data for training and testing
    X_train_ind, X_test_ind, y_train_ind, y_test_ind = split_data(df_ind_prepared)
    X_train_sys, X_test_sys, y_train_sys, y_test_sys = split_data(df_sys_prepared)

    preprocessor_ind = create_preprocessor(df_ind_prepared)
    preprocessor_sys = create_preprocessor(df_sys_prepared)


    # Define the approach
    approaches = {
        'individual_approach': (X_train_ind, X_test_ind, y_train_ind, y_test_ind, preprocessor_ind),
        'systematic_approach': (X_train_sys, X_test_sys, y_train_sys, y_test_sys, preprocessor_sys)
    }

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

    for approach_name, (X_train, X_test, y_train, y_test, preprocessor) in approaches.items():
        # Fit the preprocessor
        preprocessor.fit(X_train)
        # Run pipeline for each model
        for model_name, (model, param_grid) in models.items():
            print(f"Performing Grid Search for {approach_name} {model_name}...")
            best_model = optimize_hyperparameters(model, param_grid, X_train, y_train, preprocessor)
            print(f"Evaluating {approach_name} {model_name}...")
            evaluate_models(best_model, X_test, y_test)
            X_train_transformed = preprocessor.transform(X_train)
            
            # Get feature names after preprocessing
            feature_names = preprocessor.get_feature_names_out()
            plot_feature_importances(best_model, X_train_transformed, feature_names)
            
            save_model(best_model, approach_name, model_name)



if __name__ == "__main__":
    main()