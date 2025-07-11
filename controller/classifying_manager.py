import requests

from data.loader import DataLoader


class ClassifyingManager:

    def send_new_data_for_classifying(self, new_data: dict):
        """Classify new user-provided data using the current core."""
        # Classifier.classify_record(new_data, self.current_model)
        try:
            response = requests.post("http://127.0.0.1:8000/classify", json=new_data)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            if data.get("status") == "success":
                print(f"Predicted Class: {data.get('predicted_class')}")
            else:
                print(f"Error: {data.get('message')}")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")





    def get_data_from_user_for_classifying(self):
        """Get from user values for each feature column and return as a dict."""
        df = DataLoader.load_data_by_file_path()
        user_data = {}
        try:
            for col in df.columns[:-1]:
                unique_vals = df[col].unique()
                print(f"Select a value for '{col}':")

                for idx, val in enumerate(unique_vals, 1):
                    print(f"  {idx}. {val}")  # print every columns values.

                while True:
                    try:
                        choice = int(input(f"Enter the number (1-{len(unique_vals)}): "))
                        if 1 <= choice <= len(unique_vals):
                            user_data[col] = str(unique_vals[choice - 1])
                            break
                        else:
                            print("Invalid choice. Please try again.")
                    except ValueError:
                        print("Please enter a valid number.")
            print("\nYou entered:", user_data)
            return user_data
        except ValueError as e:
            print('An error accord', e)