import json
import requests

from core.classifier import Classifier
from data.loader import DataLoader
from core.trainer import Trainer


class TrainingManager:

    def __init__(self):
        self.model_builder = Trainer()
        self.all_models_file_paths = []

    def process_new_model(self):
        """Get dataset from user, split it, send it for training, and save the model."""
        data_set = DataLoader.load_data_by_file_path()
        training_df, testing_df = self.split_the_data_set(data_set)
        model_name = self.get_name_for_model()
        training_data = training_df.to_dict(orient="records")
        self.train_model_via_server(training_data, model_name)
        # model = self.train_new_model(training_df, model_name)
        # check = self.check_that_model_doesent_exists(model)
        # if check:
        #     path =  self.save_model_to_file(model_name, model)
        #     self.all_models_file_paths.append(path)
        #     print(f"\nModel '{model_name}' was built and saved successfully.")
        #     self.check_model_accuracy(model_name, model, testing_df)
        #
        # else:
        #     print('Model on this data already exists.')
        #     return

    def train_new_model(self, data, name):
        model = self.model_builder.build_model(data, name)
        return model

    def train_model_via_server(self, training_data, model_name):
        """
        Send training data and model name to the server for training and return the trained model.
        """
        url = "http://127.0.0.1:8000/train"
        payload = {
            "model_name": model_name,
            "data": training_data
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                model = data.get("model")
                print("Model trained successfully via server.")
                return model
            else:
                print("Server error:", data.get("message"))
        except requests.RequestException as e:
            print("Request failed:", e)
        except OSError as e:
            print("File error:", e)
        return None


    @staticmethod
    def save_model_to_file(model_name, model):
        """Save the model to a json file, and return the path of the file."""
        try:
            with open(f"models/{model_name}.json", "w", encoding="utf-8") as f:
                json.dump(model, f, indent=4)

            return f"models/{model_name}.json"
        except OSError as e:
            print('An error occurred while saving model to file: \n', e)


    @staticmethod
    def get_name_for_model():
        model_name = input('Enter the name you want to call the model: ')
        return model_name


    def check_that_model_doesent_exists(self, test_model):
        try:
            for file_path in self.all_models_file_paths:
               with open(file_path, 'r') as f:
                   model = json.load(f)
               if test_model == model:
                    return False

            return True
        except OSError as e:
           print('An error occurred while reading the models from file: \n', e)


    @staticmethod
    def check_model_accuracy(model_name, model, test_df):
        """Evaluate the core's accuracy on the test set and print the result."""
        correct = 0
        total = len(test_df)
        label_col = test_df.columns[-1]  # assuming last column is the label

        for _, row in test_df.iterrows():
            features = row.drop(label_col).to_dict()
            actual = row[label_col]
            predicted = Classifier.classify_record(features, model_name, model)
            if predicted == actual:
                correct += 1

        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"Model Accuracy: {accuracy:.2f}%\n")
        return accuracy


    @staticmethod
    def split_the_data_set(df):
        """
        Randomly split the DataFrame into 70% training and 30% testing sets.
        """
        df_shuffled = df.sample(frac=1, random_state=420).reset_index(drop=True)  # Shuffle the DataFrame
        split_index = int(len(df_shuffled) * 0.70)
        training_df = df_shuffled.iloc[:split_index]
        checking_df = df_shuffled.iloc[split_index:]
        return training_df, checking_df