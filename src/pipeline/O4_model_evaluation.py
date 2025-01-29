import numpy as np
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


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

def analyze_predictions_for_lstm(predictions_df):
    """
    Vergleicht Vorhersagen mit den echten Werten.
    """
    mae = mean_absolute_error(predictions_df['fantasy_points'], predictions_df['predicted_fantasy_points'])
    mse = mean_squared_error(predictions_df['fantasy_points'], predictions_df['predicted_fantasy_points'])
    r2 = r2_score(predictions_df['fantasy_points'], predictions_df['predicted_fantasy_points'])
    
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")
    return predictions_df

