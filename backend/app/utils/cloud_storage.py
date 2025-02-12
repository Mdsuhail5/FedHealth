from google.cloud import storage  # or boto3 for AWS
from ..config import settings
import os

class CloudStorage:
    def __init__(self):
        if settings.cloud_storage_enabled:
            self.client = storage.Client()
            self.bucket = self.client.bucket(settings.cloud_storage_bucket)
    
    def upload_model(self, local_path):
        if settings.cloud_storage_enabled:
            blob_name = os.path.basename(local_path)
            blob = self.bucket.blob(f"models/{blob_name}")
            blob.upload_from_filename(local_path)
            return blob.public_url
        return None
    
    def download_model(self, model_name, local_path):
        if settings.cloud_storage_enabled:
            blob = self.bucket.blob(f"models/{model_name}")
            blob.download_to_filename(local_path)
            return local_path
        return None 