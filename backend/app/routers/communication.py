from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import json

router = APIRouter()

class ModelUpdate(BaseModel):
    client_id: str
    round_number: int
    weights: Dict[str, List[float]]

@router.post("/send-update")
async def receive_model_update(update: ModelUpdate):
    """Receive model updates from clients"""
    try:
        # Process model update
        return {"status": "Update received successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get-global-model/{round_number}")
async def get_global_model(round_number: int):
    """Send global model to clients"""
    try:
        # Send current global model
        return {
            "round": round_number,
            "model_weights": global_model.state_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 