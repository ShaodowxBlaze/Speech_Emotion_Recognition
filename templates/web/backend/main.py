from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .services.predictor import EmotionPredictor
import uvicorn

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = EmotionPredictor(model_path="data/models/emotion_model.pt")

@app.post("/api/predict-emotion")
async def predict_emotion(audio: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_path = f"temp_{audio.filename}"
    with open(temp_path, "wb") as buffer:
        content = await audio.read()
        buffer.write(content)
    
    # Get predictions
    try:
        results = predictor.predict(temp_path)
        return results
    finally:
        import os
        os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)