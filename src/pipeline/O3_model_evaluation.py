import numpy as np
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr


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



def plot_feature_importances(model, X_train):
    model_name = model.named_steps['model'].__class__.__name__
    print(f"Plotting feature importances for model: {model_name}")

    # Plot feature importances using SHAP
    explainer = shap.Explainer(model.named_steps['model'], X_train)
    shap_values = explainer(X_train)

    # Summary plot
    shap.summary_plot(shap_values, X_train)

    # Bar plot
    shap.summary_plot(shap_values, X_train, plot_type="bar")

    print("\n")
