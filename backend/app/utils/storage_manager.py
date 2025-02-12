import os
import torch
from datetime import datetime
from ..config import settings

class StorageManager:
    def __init__(self):
        self.local_path = settings.model_storage_path
        self.backup_path = settings.backup_storage_path
        os.makedirs(self.local_path, exist_ok=True)
        os.makedirs(self.backup_path, exist_ok=True)
    
    def save_model(self, model, model_type, client_id=None):
        # Local save
        local_path = self._save_local(model, model_type, client_id)
        
        # Cloud save if enabled
        if settings.cloud_storage_enabled:
            self._save_cloud(local_path)
        
        return local_path
    
    def _save_local(self, model, model_type, client_id=None):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if model_type == 'global':
            filename = f'global_model_{timestamp}.pt'
        else:
            filename = f'client_{client_id}_{timestamp}.pt'
            
        filepath = os.path.join(self.local_path, filename)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'timestamp': timestamp,
            'type': model_type,
            'client_id': client_id
        }, filepath)
        
        # Create backup
        backup_path = os.path.join(self.backup_path, filename)
        torch.save({
            'model_state_dict': model.state_dict(),
            'timestamp': timestamp,
            'type': model_type,
            'client_id': client_id
        }, backup_path)
        
        return filepath
    
    def _save_cloud(self, local_path):
        if settings.cloud_storage_enabled:
            # Add your cloud storage logic here
            # Example: AWS S3, Google Cloud Storage, etc.
            pass 