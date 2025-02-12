import sqlite3
import json
from datetime import datetime
import os

class Database:
    def __init__(self):
        self.db_path = 'federated_learning.db'
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create tables
        c.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_type TEXT,
            client_id TEXT,
            timestamp TEXT,
            filepath TEXT,
            accuracy REAL,
            metrics TEXT
        )''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS training_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER,
            epoch INTEGER,
            loss REAL,
            accuracy REAL,
            val_accuracy REAL,
            FOREIGN KEY(model_id) REFERENCES models(id)
        )''')
        
        conn.commit()
        conn.close()
    
    def save_model_info(self, model_type, client_id, filepath, accuracy, metrics):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
        INSERT INTO models (model_type, client_id, timestamp, filepath, accuracy, metrics)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            model_type,
            client_id,
            datetime.now().isoformat(),
            filepath,
            accuracy,
            json.dumps(metrics)
        ))
        
        model_id = c.lastrowid
        conn.commit()
        conn.close()
        return model_id
    
    def save_training_history(self, model_id, history):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        for epoch, metrics in enumerate(zip(
            history['loss'],
            history['accuracy'],
            history['val_accuracy']
        )):
            loss, acc, val_acc = metrics
            c.execute('''
            INSERT INTO training_history (model_id, epoch, loss, accuracy, val_accuracy)
            VALUES (?, ?, ?, ?, ?)
            ''', (model_id, epoch, loss, acc, val_acc))
        
        conn.commit()
        conn.close()
    
    def get_model_history(self, client_id=None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        if client_id:
            c.execute('''
            SELECT * FROM models WHERE client_id = ? ORDER BY timestamp DESC
            ''', (client_id,))
        else:
            c.execute('SELECT * FROM models ORDER BY timestamp DESC')
        
        models = c.fetchall()
        conn.close()
        return models
    
    def get_latest_global_model(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
        SELECT * FROM models 
        WHERE model_type = 'global' 
        ORDER BY timestamp DESC 
        LIMIT 1
        ''')
        
        model = c.fetchone()
        conn.close()
        return model 