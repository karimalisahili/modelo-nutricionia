from fastapi import FastAPI, Form, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Load the trained model
model = load_model('food_classifier_model.h5')

# Define the target categories
target_categories = ['hamburger', 'pizza']

# Nutritional information per 100 grams
food_nutrition = {
    'hamburger': {'mass':100, 'calories': 295, 'fat': 14, 'protein': 17, 'description': 'Una hamburguesa es un sándwich que consiste en una o más hamburguesas cocidas de carne molida, generalmente de res, colocadas dentro de una rebanada de pan.'},
    'pizza': {'mass':100, 'calories': 266, 'fat': 10, 'protein': 11, 'description': 'La pizza consiste en una base circular de masa de pan horneada a la que se le añaden ingredientes variados.'}
}

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
origins = [os.getenv('CORS_ORIGIN')]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a Pydantic model for the request body
class ImageRequest(BaseModel):
    weight: float  # Weight of the food in grams

# Helper function to preprocess the image
def preprocess_image(image: Image.Image, target_size: tuple) -> np.ndarray:
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Helper function to calculate nutrition based on weight
def calculate_nutrition(food_name: str, weight: str = Form(...)):
    weight = float(weight)
    print(weight)
    nutrition = food_nutrition.get(food_name)
    if not nutrition:
        raise ValueError(f"Nutrition data not found for {food_name}")
    
    # Calculate based on the weight (rule of three)
    calories = (nutrition['calories'] * weight) / nutrition['mass']
    fat = (nutrition['fat'] * weight) / nutrition['mass']
    protein = (nutrition['protein'] * weight) / nutrition['mass']
    description = nutrition['description']
    
    return {
        'food_name': food_name,
        'calories': calories,
        'fat': fat,
        'protein': protein,
        'description': description
    }

# Endpoint to receive image and return prediction
@app.post("/predict/")
async def predict_food(image: UploadFile = File(...)):
    try:
        # Read the image file into memory
        image_data = await image.read()
        image = Image.open(BytesIO(image_data))

        # Preprocess the image
        target_size = (222, 222)  # Update this to match your model's expected input size
        processed_image = preprocess_image(image, target_size)

        # Make prediction
        predictions = model.predict(processed_image)
        print(f"Raw predictions: {predictions}")

        # Determine the label based on the prediction
        confidence = predictions[0][0]
        if 0.05 < confidence < 0.95:
            label = 'Not recognized'
        else:
            label = 'pizza' if confidence > 0.5 else 'hamburger'

        print(f"Prediction: {label}")

        # Return the prediction as JSON
        return JSONResponse(content={"food_name": label})

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
