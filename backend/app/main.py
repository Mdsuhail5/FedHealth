from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import client, communication
from app.config import settings

app = FastAPI(title="Federated Health Classification")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(client.router, prefix="/client", tags=["client"])
app.include_router(communication.router, prefix="/comm", tags=["communication"])

@app.get("/")
async def root():
    return {"message": "Federated Learning Health Classification System"} 