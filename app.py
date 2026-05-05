from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os
from model_inference import predict_image_int32, get_snake_info
from GPT_client import get_snake_description, LLM_snake_identifier
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="Snake Identification API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"]
    ,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def helper():
    return "snake"

@app.post("/identify-snake")
async def identify_snake(image: UploadFile = File(...)):
    """Endpoint 1: Identify snake species from image"""
    image_path = os.path.join(UPLOAD_DIR, image.filename)
    with open(image_path, "wb") as f:
        f.write(await image.read())

    pred_class, species_name, confidence = predict_image_int32(image_path)
    venom_status, antivenom, habitat = get_snake_info(pred_class)
    about_snake = get_snake_description(species_name)
    
    

    result = {
        "Image": image.filename,
        "Identification Confidence": f"{confidence:.2f}%",
        "Predicted Species": species_name,
        "Venomous status": venom_status,
        "About This Snake": about_snake,
        "Habitat & Distribution": habitat,
        "Danger Level": "High" if venom_status == "Venomous" else "Low",
        "Recommended Antivenom": antivenom,
    }

    return JSONResponse(result)

@app.post("/describe-snake")
async def description_base_snake_identify(text: str = Form(...)):
    """Endpoint 2: Describe snake by name or text"""
    return LLM_snake_identifier(text)