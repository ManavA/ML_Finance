# src/training/metrics.py
def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    # Flatten arrays if needed
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # MSE
    mse = np.mean((predictions - targets) ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(np.abs(predictions - targets))
    
    # MAPE
    mask = targets != 0
    mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
    
    # R2 Score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Direction accuracy (for price movement)
    if len(predictions) > 1:
        pred_direction = np.diff(predictions) > 0
        true_direction = np.diff(targets) > 0
        direction_accuracy = np.mean(pred_direction == true_direction)
    else:
        direction_accuracy = 0
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(r2),
        'direction_accuracy': float(direction_accuracy)
    }