# src/app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prediction_service import EmotionPredictor
import uvicorn
from pathlib import Path
import shutil

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
try:
    predictor = EmotionPredictor()
except Exception as e:
    print(f"Error initializing predictor: {str(e)}")
    raise

@app.post("/predict")
async def predict_emotion(audio: UploadFile = File(...)):
    # Create temp directory if it doesn't exist
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Save uploaded file
    temp_path = temp_dir / audio.filename
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        # Get predictions
        predictions = predictor.predict(str(temp_path))
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)