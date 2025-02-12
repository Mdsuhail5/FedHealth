import torch
import os
from datetime import datetime

def save_model(model, model_type, client_id=None):
    """Save model to disk"""
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if model_type == 'global':
        filename = f'models/global_model_{timestamp}.pt'
    else:
        filename = f'models/client_{client_id}_{timestamp}.pt'
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'timestamp': timestamp,
        'type': model_type,
        'client_id': client_id
    }, filename)
    return filename

def load_model(filepath, model_class):
    """Load model from disk"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No model found at {filepath}")
    
    checkpoint = torch.load(filepath)
    model = model_class(input_size=4)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint 