import json
from datetime import datetime
import os

class ExperimentTracker:
    def __init__(self):
        self.experiments_dir = 'experiments'
        os.makedirs(self.experiments_dir, exist_ok=True)
        
    def log_experiment(self, experiment_data):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{self.experiments_dir}/experiment_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(experiment_data, f, indent=4)
        
        return filename
    
    def load_experiment(self, filename):
        with open(filename, 'r') as f:
            return json.load(f) 