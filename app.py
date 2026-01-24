from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import pickle
import uvicorn


app = FastAPI(title="API de clasificación de imágenes", version="1.0")
#middleware facilita la comunicación entre frontend y backend si están en  distintos puertos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = tf.keras.models.load_model("modelo_frutas.keras")
    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
except Exception as e:
    print(f"✗ Error cargando modelo")
    model = None
    encoder = None

@app.get("/")
def read_root():
    return FileResponse("index.html", media_type="html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)): #Predice el tipo de fruta

    if model is None or encoder is None:
        return {"error": "Modelo no cargado"}
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))
        
        img_array = np.array(image, dtype='float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction))
        etiqueta = encoder.inverse_transform([predicted_class])[0]
        
        return {
            "categoria": etiqueta
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    return {"status": "online", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)