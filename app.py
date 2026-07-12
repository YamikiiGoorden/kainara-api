import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
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

print("Building model architecture...")

# Bangun ulang arsitektur PERSIS sesuai model.summary():
# sequential -> rescaling -> resnet50 -> global_avg_pool -> dense(256) -> dropout -> dense(20)

base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights=None,          # jangan load imagenet, nanti ketimpa weights hasil training
    input_shape=(224, 224, 3)
)

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(CLASS_NAMES), activation='softmax')
])

print("Loading weights...")
model.load_weights("batik_resnet50.weights.h5")
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

        prediction = model.predict(img_array, verbose=0)[0]
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
