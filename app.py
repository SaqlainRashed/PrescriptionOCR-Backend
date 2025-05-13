from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from medicine_service import MedicineService
from pathlib import Path
import tempfile
import os
import uvicorn
from fastapi import BackgroundTasks
import asyncio
import logging

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize service
DB_PATH = "data/cleaned_expanded_drug_dataset.csv"  # Update this path
service = MedicineService(DB_PATH)

app = FastAPI(
    title="Prescription OCR API",
    description="API for extracting text from prescriptions and suggesting medicines",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/test")
async def test_endpoint():
    """Basic health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "message": "API is working!",
            "endpoints": {
                "test": "GET /test",
                "process": "POST /process-prescription",
                "health": "GET /health"
            }
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database_loaded": os.path.exists(DB_PATH),
        "service_ready": True
    }

# NEW CODE FOR MULTI MEDICINE DATA
@app.post("/process-prescription")
async def process_prescription(
    file: UploadFile = File(...),
    mode: str = Form("SINGLE_MED")  # Get the mode from form data
):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "Only image files are allowed")
            
        return await asyncio.wait_for(_process_image(file, mode), timeout=30.0)
    except asyncio.TimeoutError:
        raise HTTPException(504, "Processing timeout")
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(500, "Failed to process prescription")

async def _process_image(file: UploadFile, mode: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_path = temp_file.name
        contents = await file.read()
        temp_file.write(contents)
    
    extracted_text, suggestions = service.process_prescription(temp_path, mode)
    os.unlink(temp_path)
    
    return {
        "extracted_text": extracted_text,
        "suggestions": suggestions,
        "mode": mode
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)