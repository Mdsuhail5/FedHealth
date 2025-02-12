from fastapi import APIRouter, File, UploadFile, HTTPException
import pandas as pd
import torch
from ..models.health_model import HealthModel
from ..utils.data_processor import DataProcessor
import io
from typing import List
import numpy as np
from ..utils.model_utils import save_model, load_model
from ..utils.metrics import calculate_metrics
from ..utils.experiment_tracker import ExperimentTracker
from ..config import settings
from datetime import datetime
from ..utils.database import Database

router = APIRouter()
data_processor = DataProcessor()

# Global model instance
global_model = HealthModel(input_size=4)
trained_clients = set()

# Store client data
client_data = {}

# Initialize experiment tracker
experiment_tracker = ExperimentTracker()

# Initialize database
db = Database()

@router.post("/upload-data/")
async def upload_data(file: UploadFile = File(...)):
    try:
        # Read file content
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Process data
        X_train, X_test, y_train, y_test = data_processor.process_data(df)
        
        # Generate client ID
        client_id = f"client_{len(client_data)}"
        
        # Store processed data
        client_data[client_id] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': data_processor.scaler,
            'model': None
        }
        
        return {
            "message": "Data processed successfully",
            "client_id": client_id,
            "train_shape": X_train.shape,
            "test_shape": X_test.shape,
            "healthy_count": int(df['healthy'].sum()),
            "total_samples": len(df)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/train/{client_id}")
async def train_client_model(client_id: str):
    try:
        if client_id not in client_data:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Get client's data
        client = client_data[client_id]
        X_train = torch.FloatTensor(client['X_train'])
        y_train = torch.FloatTensor(client['y_train']).reshape(-1, 1)
        X_test = torch.FloatTensor(client['X_test'])
        y_test = torch.FloatTensor(client['y_test']).reshape(-1, 1)
        
        # Initialize model
        model = HealthModel(input_size=settings.model_input_size)
        if hasattr(global_model, 'state_dict'):
            model.load_state_dict(global_model.state_dict())
        
        # Training setup
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=settings.learning_rate, 
            weight_decay=settings.weight_decay
        )
        criterion = torch.nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        # Training loop
        history = {'loss': [], 'accuracy': [], 'val_accuracy': [], 'metrics': []}
        best_val_acc = 0
        best_model = None
        
        for epoch in range(settings.training_epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            # Mini-batch training
            for i in range(0, len(X_train), settings.batch_size):
                batch_X = X_train[i:i+settings.batch_size]
                batch_y = y_train[i:i+settings.batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                predictions = (outputs >= 0.5).float()
                correct += (predictions == batch_y).sum().item()
                total += len(batch_y)
            
            # Calculate metrics
            epoch_loss = total_loss / (len(X_train) / settings.batch_size)
            train_accuracy = correct / total
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_preds = (val_outputs >= 0.5).float()
                val_metrics = calculate_metrics(
                    y_test.numpy(), 
                    val_preds.numpy(), 
                    val_outputs.numpy()
                )
            
            # Store metrics
            history['loss'].append(epoch_loss)
            history['accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['metrics'].append(val_metrics)
            
            # Update learning rate
            scheduler.step(epoch_loss)
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_model = model.state_dict()
        
        # Use best model
        model.load_state_dict(best_model)
        client_data[client_id]['model'] = model
        trained_clients.add(client_id)
        
        # Save model
        model_path = save_model(model, 'client', client_id)
        
        # Save to database
        model_id = db.save_model_info(
            model_type='client',
            client_id=client_id,
            filepath=model_path,
            accuracy=best_val_acc,
            metrics=history['metrics'][-1]
        )
        
        # Save training history
        db.save_training_history(model_id, history)
        
        return {
            "message": "Training completed successfully",
            "training_history": history,
            "best_accuracy": best_val_acc,
            "model_path": model_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict/{client_id}/test")
async def predict_test_data(client_id: str):
    try:
        if client_id not in client_data:
            raise HTTPException(status_code=404, detail="Client not found")
        
        client = client_data[client_id]
        if client['model'] is None:
            raise HTTPException(status_code=400, detail="Model not trained yet")
        
        # Make predictions on test data
        X_test = torch.FloatTensor(client['X_test'])
        y_test = torch.FloatTensor(client['y_test']).reshape(-1, 1)
        
        with torch.no_grad():
            client['model'].eval()
            outputs = client['model'](X_test)
            predictions = (outputs >= 0.5).float()
            accuracy = (predictions == y_test).float().mean().item()
        
        return {
            "accuracy": accuracy,
            "predictions": predictions.numpy().tolist(),
            "actual_values": y_test.numpy().tolist(),
            "probabilities": outputs.numpy().tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/aggregate/")
async def aggregate_models():
    try:
        if len(trained_clients) < 1:
            raise HTTPException(
                status_code=400, 
                detail="No trained models available for aggregation"
            )
        
        # Collect weights and calculate total samples
        all_weights = []
        sample_sizes = []
        
        for client_id in trained_clients:
            client = client_data[client_id]
            if client['model'] is not None:
                # Clone and ensure correct dimensions
                weights = {}
                for k, v in client['model'].state_dict().items():
                    weights[k] = v.clone().detach().float()
                all_weights.append(weights)
                sample_sizes.append(len(client['y_train']))
        
        total_samples = sum(sample_sizes)
        sample_weights = [size/total_samples for size in sample_sizes]
        
        # Weighted average of the weights (FedAvg algorithm)
        averaged_weights = {}
        for key in all_weights[0].keys():
            # Handle weights and biases separately
            if 'weight' in key:
                stacked = torch.stack([w[key] for w in all_weights])
                averaged_weights[key] = torch.sum(
                    stacked * torch.tensor(sample_weights).view(-1, 1, 1), 
                    dim=0
                )
            else:  # bias
                stacked = torch.stack([w[key] for w in all_weights])
                averaged_weights[key] = torch.sum(
                    stacked * torch.tensor(sample_weights).view(-1, 1), 
                    dim=0
                )
        
        # Create new global model instance
        new_global_model = HealthModel(input_size=4)
        
        # Verify shapes before loading
        current_state = new_global_model.state_dict()
        for key in averaged_weights:
            if averaged_weights[key].shape != current_state[key].shape:
                raise ValueError(
                    f"Shape mismatch for {key}: "
                    f"averaged={averaged_weights[key].shape}, "
                    f"expected={current_state[key].shape}"
                )
        
        # Load verified weights
        new_global_model.load_state_dict(averaged_weights)
        
        # Evaluate new global model
        results = []
        total_correct = 0
        total_samples = 0
        
        for client_id in trained_clients:
            client = client_data[client_id]
            X_test = torch.FloatTensor(client['X_test'])
            y_test = torch.FloatTensor(client['y_test']).reshape(-1, 1)
            
            with torch.no_grad():
                new_global_model.eval()
                outputs = new_global_model(X_test)
                predictions = (outputs >= 0.5).float()
                correct = (predictions == y_test).sum().item()
                accuracy = correct / len(y_test)
                
                total_correct += correct
                total_samples += len(y_test)
                
                results.append({
                    'client_id': client_id,
                    'accuracy': accuracy,
                    'test_samples': len(y_test)
                })
        
        # Update global model
        global global_model
        global_model = new_global_model
        
        # Update client models with new global weights
        for client_id in trained_clients:
            client = client_data[client_id]
            client['model'].load_state_dict(averaged_weights)
        
        global_accuracy = total_correct / total_samples
        
        # Save global model
        model_path = save_model(global_model, 'global')
        
        # Log experiment
        experiment_data = {
            'timestamp': datetime.now().isoformat(),
            'num_clients': len(trained_clients),
            'global_accuracy': global_accuracy,
            'client_results': results,
            'model_path': model_path,
            'settings': settings.dict()
        }
        experiment_path = experiment_tracker.log_experiment(experiment_data)
        
        return {
            "message": "Global model updated successfully",
            "num_clients": len(trained_clients),
            "global_accuracy": global_accuracy,
            "client_results": results,
            "model_path": model_path,
            "experiment_path": experiment_path
        }
    except Exception as e:
        print(f"Aggregation error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/global-model-status")
async def get_global_model_status():
    return {
        "num_clients": len(trained_clients),
        "clients": list(trained_clients)
    }

@router.get("/global-model-metrics")
async def get_global_model_metrics():
    try:
        if len(trained_clients) < 1:
            return {
                "status": "No trained clients",
                "num_clients": 0
            }
        
        client_metrics = []
        total_test_samples = 0
        total_weighted_accuracy = 0
        
        for client_id in trained_clients:
            client = client_data[client_id]
            X_test = torch.FloatTensor(client['X_test'])
            y_test = torch.FloatTensor(client['y_test']).reshape(-1, 1)
            
            # Test local model
            with torch.no_grad():
                client['model'].eval()
                local_outputs = client['model'](X_test)
                local_preds = (local_outputs >= 0.5).float()
                local_acc = (local_preds == y_test).float().mean().item()
            
            # Test global model
            with torch.no_grad():
                global_model.eval()
                global_outputs = global_model(X_test)
                global_preds = (global_outputs >= 0.5).float()
                global_acc = (global_preds == y_test).float().mean().item()
            
            metrics = {
                "client_id": client_id,
                "test_samples": len(y_test),
                "local_accuracy": local_acc,
                "global_accuracy": global_acc
            }
            client_metrics.append(metrics)
            
            total_test_samples += len(y_test)
            total_weighted_accuracy += global_acc * len(y_test)
        
        return {
            "num_clients": len(trained_clients),
            "total_test_samples": total_test_samples,
            "average_global_accuracy": total_weighted_accuracy / total_test_samples,
            "client_metrics": client_metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{client_id}")
async def get_client_history(client_id: str):
    try:
        history = db.get_model_history(client_id)
        return {
            "client_id": client_id,
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 