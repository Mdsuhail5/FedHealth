from pydantic import BaseSettings

class Settings(BaseSettings):
    model_input_size: int = 4
    training_epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    test_size: float = 0.2
    random_seed: int = 42
    model_storage_path: str = "models"
    backup_storage_path: str = "backups"
    cloud_storage_enabled: bool = False
    cloud_storage_bucket: str = ""
    
    class Config:
        env_file = ".env"

settings = Settings() 