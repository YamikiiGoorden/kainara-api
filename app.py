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

        # PENTING:
        # - Input harus raw pixel 0-255 (float32), JANGAN dibagi 255 di sini,
        #   karena model sudah punya layer Rescaling(1./255) built-in.
        # - Output model SUDAH melalui activation='softmax' di layer terakhir,
        #   jadi JANGAN panggil tf.nn.softmax() lagi di server (itu bug lama
        #   yang bikin confidence selalu mepet ~5%, alias hampir random).
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        input_tensor = tf.constant(img_array)

        result = infer(input_layer_1=input_tensor)
        prediction = result["output_0"].numpy()[0]

        # Output sudah berupa probabilitas (sum ~1), pakai langsung
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
