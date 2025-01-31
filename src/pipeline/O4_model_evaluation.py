import numpy as np
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns

def evaluate_models_for_lr_and_xgb(model, X_test, y_test):
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

def plot_feature_importances_for_lr_and_xgb(model, X_train, feature_names):
    model_name = model.named_steps['model'].__class__.__name__
    print(f"Plotting feature importances for model: {model_name}")

    # Plot feature importances using SHAP
    explainer = shap.Explainer(model.named_steps['model'], X_train)
    shap_values = explainer(X_train)

    # Summary plot
    shap.summary_plot(shap_values, features=X_train, feature_names=feature_names)

    # Bar plot
    shap.summary_plot(shap_values, features=X_train, feature_names=feature_names, plot_type="bar")

    print("\n")

def plot_results_for_lr_and_xgb(model, X_test, y_test):
    y_pred = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Fantasy Points')
    plt.grid(True)
    plt.show()

def analyze_model_performance_for_lstm(history):
    """
    Analysiert die Modellperformance.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_predictions_for_lstm(predictions_df):
    """
    Vergleicht Vorhersagen mit den echten Werten.
    """
    mae = mean_absolute_error(predictions_df['fantasy_points'], predictions_df['predicted_fantasy_points'])
    mse = mean_squared_error(predictions_df['fantasy_points'], predictions_df['predicted_fantasy_points'])
    rmse = np.sqrt(mse)
    r2 = r2_score(predictions_df['fantasy_points'], predictions_df['predicted_fantasy_points'])
    spearman_corr = spearmanr(predictions_df['fantasy_points'], predictions_df['predicted_fantasy_points'])[0]
    
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}, Spearman Rank Correlation: {spearman_corr:.2f}")
    return predictions_df

def plot_feature_importance_for_lstm(model, X_test, y_test, feature_names):
    """
    Analyze and visualize feature importance using multiple methods
    """
    import matplotlib.pyplot as plt
    
    # 1. Create baseline predictions
    baseline_preds = model.predict(X_test)
    baseline_mse = mean_squared_error(y_test, baseline_preds)
    
    # 2. Feature Importance through Permutation
    importances = []
    
    for i in range(X_test.shape[2]):  # iterate through features
        X_test_temp = X_test.copy()
        
        # Permute the i-th feature
        X_test_temp[:, :, i] = np.random.permutation(X_test_temp[:, :, i])
        
        # Get predictions with permuted feature
        perm_preds = model.predict(X_test_temp, verbose=0)
        perm_mse = mean_squared_error(y_test, perm_preds)
        
        # Calculate importance
        importance = perm_mse - baseline_mse
        importances.append(importance)
    
    # Create DataFrame of importances
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Visualizations
    plt.figure(figsize=(12, 6))
    
    # Plot feature importances
    sns.barplot(data=importance_df.head(20), x='importance', y='feature')
    plt.title('Top 20 Most Important Features')
    plt.xlabel('Importance (Increase in MSE when feature is permuted)')
    
    plt.tight_layout()
    plt.show()
    
    # Print top features
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    
    return importance_df


def plot_results_for_lstm(predictions_df):
    """
    Plots a scatter plot comparing actual and predicted target values for LSTM model.
    """
    y_test = predictions_df['fantasy_points']
    y_pred = predictions_df['predicted_fantasy_points']
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Fantasy Points')
    plt.grid(True)
    plt.show()
