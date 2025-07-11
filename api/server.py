from fastapi import FastAPI, Request
from typing import Dict, Any
import pandas as pd
import os
import json

from core.trainer import Trainer
from core.classifier import Classifier


app = FastAPI()

# Global variable to store the trained model.
# In a production environment, consider more robust model management (e.g., database, file system).
trained_model = None
model_name = ''
@app.post('/train')
async def train_model(request: Request):
    """
    Trains a Naive Bayes model using JSON data (list of dicts).
    The trained model is stored in memory for subsequent classifications.
    Also saves the model as a JSON file in the models/ directory, if it doesn't already exist.
    """
    global trained_model
    global model_name
    try:
        body = await request.json()
        model_name = body["model_name"]
        data = body["data"]
        df = pd.DataFrame(data)
        # Initialize and build the model
        trainer = Trainer()
        trained_model = trainer.build_model(df, model_name)

        # Prepare file path
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        model_file_path = os.path.join(models_dir, f"{model_name}.json")

        # Check if model already exists
        if os.path.exists(model_file_path):
            return {"message": f"Model '{model_name}' already exists.", "status": "error"}

        # Save the model to a JSON file
        with open(model_file_path, 'w', encoding='utf-8') as f:
            json.dump(trained_model, f, ensure_ascii=False, indent=4)

        return {"model": trained_model, "status": 'success'}
    except Exception as e:
        return {"message": f"An error occurred during training: {e}", "status": "error"}


@app.post('/classify')
async def classify_data(new_example: Dict[str, Any]):
        """
        Classifies a new data example using the currently trained Naive Bayes model.
        The model must be trained via the /train endpoint before classification can occur.
        """
        global trained_model

        if trained_model is None:
            return {"message": "No model has been trained yet. Please train a model first using the /train endpoint.",
                    "status": "error"}

        try:
            # Classify the new example
            predicted_class = Classifier.classify_record(new_example, model_name, trained_model)

            if predicted_class:
                return {"predicted_class": predicted_class, "status": "success"}
            else:
                return {"message": "Classification failed.", "status": "error"}
        except Exception as e:
            return {"message": f"An error occurred during classification: {e}", "status": "error"}

@app.get('/models')
async def get_all_models():
    pass

@app.get('/')
async def root():
    return {"message": "Naive Bayes Classifier API. Use /train to train a model and /classify to classify data."}

