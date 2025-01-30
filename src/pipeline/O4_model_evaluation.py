import numpy as np
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import tensorflow as tf



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

def plot_coefs_for_lr_and_xgb(approach_name, model, feature_names):
    if approach_name in ['individual_lr_approach', 'systematic_lr_approach']:
        print(f"Plotting coefficients for model: {approach_name}")
        coefs = model.named_steps['model'].coef_
        intercept = model.named_steps['model'].intercept_
        
        # Combine intercept and coefficients
        coefs = np.append(intercept, coefs)
        feature_names = ['Intercept'] + feature_names

        # Get top 20 coefficients excluding intercept
        top_indices = np.argsort(np.abs(coefs[1:]))[-19:] + 1
        top_coefs = coefs[top_indices]
        top_features = [feature_names[i] for i in top_indices if i < len(feature_names)]
        # Add intercept to the top features and coefficients
        top_coefs = np.insert(top_coefs, 0, intercept)
        top_features.insert(0, 'Intercept')

        plt.figure(figsize=(10, 6))
        bars = plt.barh(top_features, top_coefs, color='b', align='center')
        bars[0].set_color('r')  # Set intercept bar color to red
        plt.xlabel('Coefficient Value')
        plt.title('Top 20 Coefficients')
        plt.grid(True)
        plt.show()
    

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

    # Print out complete training and validation loss
    print("Training Loss:", history.history['loss'])
    print("Validation Loss:", history.history['val_loss'])

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


def plot_shap_analysis_for_lstm(model, X_train):
    """
    Führt eine SHAP-Analyse für ein LSTM-Modell durch und gibt den entsprechenden Plot aus.

    :param model: Das trainierte LSTM-Modell
    :param X_train: Trainingsdaten als numpy-Array (3D: samples, timesteps, features)
    :param sample_size: Anzahl der zu analysierenden Stichproben
    """
    sample_size=100
    # Stichproben für die Analyse
    X_sample = X_train[:sample_size].astype(np.float32)

    # Hintergrundstichprobe für den Explainer
    background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)].astype(np.float32)

    # Explainer für neuronale Netze (stabiler als DeepExplainer)
    explainer = shap.GradientExplainer(model, background)

    # SHAP-Werte berechnen
    shap_values = explainer.shap_values(X_sample)

    # Falls Klassifikationsmodell -> erste Klasse nehmen
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # SHAP-Plot für das erste Feature über die Zeit
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[:, :, 0], X_sample[:, :, 0])
    plt.show()




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
