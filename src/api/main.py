from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from typing import Dict, List, Optional
import uvicorn
import os

from src.models.helformer_enhanced import create_helformer_for_crypto
from src.models.lstm_gru_enhanced import create_lstm_gru_for_crypto
from src.models.advanced_rl_enhanced import create_rl_agent

app = FastAPI(title="ML Finance Trading API", version="1.0.0")

# Global model storage
models = {}

class PredictionRequest(BaseModel):
    data: List[List[float]]
    model_type: str = "helformer"
    volume: Optional[List[float]] = None
    volatility: Optional[List[float]] = None
    sentiment: Optional[List[float]] = None

class PredictionResponse(BaseModel):
    predictions: List[float]
    direction: int
    confidence: float
    model_used: str

@app.on_event("startup")
async def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models on {device}")
    
    try:
        models["helformer"] = create_helformer_for_crypto(device=device)
        models["lstm_gru"] = create_lstm_gru_for_crypto(device=device)
        models["rl_td3"] = create_rl_agent("td3", state_dim=32, action_dim=3, device=device)
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")

@app.get("/")
async def root():
    return {"message": "ML Finance Trading API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        if request.model_type not in models:
            raise HTTPException(status_code=400, detail=f"Model {request.model_type} not available")
        
        model = models[request.model_type]
        
        # Convert to tensor
        data = torch.FloatTensor(request.data)
        
        # Add batch dimension if needed
        if len(data.shape) == 2:
            data = data.unsqueeze(0)
        
        with torch.no_grad():
            if request.model_type == "helformer":
                volume = torch.FloatTensor(request.volume) if request.volume else None
                volatility = torch.FloatTensor(request.volatility) if request.volatility else None
                sentiment = torch.FloatTensor(request.sentiment) if request.sentiment else None
                
                output = model(data, volume=volume, volatility=volatility, 
                             sentiment=sentiment, return_uncertainty=True)
            elif request.model_type == "lstm_gru":
                output = model(data, return_uncertainty=True)
            else:
                # RL agent
                state = data.numpy().flatten()
                action = model.select_action(state, add_noise=False)
                output = {
                    "predictions": action,
                    "direction": int(np.argmax(action)),
                    "confidence": float(np.max(np.abs(action)))
                }
                return PredictionResponse(
                    predictions=action.tolist(),
                    direction=output["direction"],
                    confidence=output["confidence"],
                    model_used=request.model_type
                )
        
        predictions = output["predictions"].cpu().numpy().tolist()
        direction = int(output["direction"].cpu().item())
        confidence = float(output.get("confidence", output["probabilities"].max()).cpu().item())
        
        return PredictionResponse(
            predictions=predictions[0] if isinstance(predictions[0], list) else predictions,
            direction=direction,
            confidence=confidence,
            model_used=request.model_type
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    return {
        "available_models": list(models.keys()),
        "total": len(models)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)