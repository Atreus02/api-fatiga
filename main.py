import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI()

# Cargar modelo .tflite
interpreter = tf.lite.Interpreter(model_path=r"model/modelo_grande.tflite")
interpreter.allocate_tensors()

# Obtener información de entradas y salidas
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Procesar imagen para .tflite
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L')  
    img = img.resize((160, 120))  # 👈 ancho=160, alto=120
    img_array = np.array(img).astype('float32') / 255.0

    # Asegurar formato (1, 120, 160, 1)
    img_array = np.expand_dims(img_array, axis=0)     # (1, 120, 160)
    img_array = np.expand_dims(img_array, axis=-1)    # (1, 120, 160, 1)
    return img_array


@app.post("/predecir")
async def predecir_fatiga(file: UploadFile = File(...)):
    contents = await file.read()
    input_data = preprocess_image(contents)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    clases = ['despierto', 'cansado', 'dormido']
    resultado = clases[np.argmax(output_data)]
    confianza = float(np.max(output_data))

    return JSONResponse(content={"estado": resultado, "confianza": confianza})

# Servir archivos estáticos

@app.get("/")
async def index():
    return FileResponse("index.html")
