import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI()

CLASS_NAMES = [
    'Batik Dayak Kalbar', 'Batik Dayak Kalmut', 'Batik Dayak Kalsel',
    'Batik Dayak Kalteng', 'Batik Dayak Kaltim', 'Batik-tujuh-rupa',
    'batik-betawi', 'batik-bokor-kencono', 'batik-buketan',
    'batik-jlamprang', 'batik-kawung', 'batik-liong',
    'batik-sidomukti', 'batik-sidomulyo', 'batik-srikaton',
    'batik-tribusono', 'batik-tuntrum', 'batik-wahyu-tumurun',
    'batik-wirasat', 'sidoluhur'
]

print("Loading model...")
# PENTING: load model Keras langsung (.h5), BUKAN SavedModel.
# tf.saved_model.load() + signatures terbukti menghasilkan prediksi salah
# untuk model ResNet50 fine-tuned ini (kemungkinan BatchNorm tidak
# dalam mode inference saat export SavedModel). model.predict() Keras
# biasa terbukti benar (99.99% confidence pada tes manual).
model = tf.keras.models.load_model("batik_resnet50.h5")
print("Model loaded!")


@app.get("/")
def root():
    return {"status": "KAINARA API aktif", "classes": len(CLASS_NAMES)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))

        # Raw pixel 0-255, JANGAN dibagi 255 di sini.
        # Layer Rescaling(1./255) sudah built-in di dalam model.
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Pakai predict() Keras biasa (bukan SavedModel signature).
        prediction = model.predict(img_array, verbose=0)[0]

        # Output sudah softmax dari layer terakhir, pakai langsung.
        score = prediction

        predicted_class = CLASS_NAMES[int(np.argmax(score))]
        confidence = float(100 * np.max(score))

        top3_idx = np.argsort(score)[::-1][:3]
        top3 = [
            {"motif": CLASS_NAMES[int(i)], "confidence": round(float(100 * score[i]), 2)}
            for i in top3_idx
        ]

        return JSONResponse({
            "motif": predicted_class,
            "confidence": round(confidence, 2),
            "top3": top3
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
