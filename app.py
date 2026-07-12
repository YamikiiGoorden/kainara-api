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
model = tf.saved_model.load("batik_savedmodel")
infer = model.signatures["serving_default"]
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
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        input_tensor = tf.constant(img_array)
        result = infer(input_tensor)

        output_key = list(result.keys())[0]
        prediction = result[output_key].numpy()[0]

        score = tf.nn.softmax(prediction).numpy()
        predicted_class = CLASS_NAMES[np.argmax(score)]
        confidence = float(100 * np.max(score))

        top3_idx = np.argsort(score)[::-1][:3]
        top3 = [{"motif": CLASS_NAMES[i], "confidence": round(float(100 * score[i]), 2)} for i in top3_idx]

        return JSONResponse({
            "motif": predicted_class,
            "confidence": round(confidence, 2),
            "top3": top3
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
