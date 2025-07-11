from fastapi import FastAPI
from typing import Dict, Any
import pandas as pd

from core.trainer import Trainer
from core.classifier import Classifier


app = FastAPI()

# Global variable to store the trained model.
# In a production environment, consider more robust model management (e.g., database, file system).
trained_model = None

@app.post('/train')
async def train_model(self, file_path: str):
        """
        Trains a Naive Bayes model using a provided CSV file path.
        The trained model is stored in memory for subsequent classifications.
        """
        global trained_model

        try:
            df = pd.read_csv(file_path)
            # Initialize and build the model
            trainer = Trainer()
            trained_model = trainer.build_model(df)

            if trained_model:
                return {"model": trained_model, "status": 'success'}
            else:
                return {"message": "Model training failed.", "status": "error"}
        except Exception as e:
            return {"message": f"An error occurred during training: {e}", "status": "error"}


@app.post('/classify')
async def classify_data(self, new_example: Dict[str, Any]):
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
            predicted_class = Classifier.classify_example(new_example, trained_model)

            if predicted_class:
                return {"predicted_class": predicted_class, "status": "success"}
            else:
                return {"message": "Classification failed.", "status": "error"}
        except Exception as e:
            return {"message": f"An error occurred during classification: {e}", "status": "error"}


@app.get('/')
async def root(self):
    return {"message": "Naive Bayes Classifier API. Use /train to train a model and /classify to classify data."}

