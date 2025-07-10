from services.UI import Ui
from model.classifier import Classifier
from data.get_raw_data import GetData
from model.process_model import ModelProcessing


class Manager:
    def __init__(self):
        self.__models = []
        self.model_builder = ModelProcessing()
        self.current_data_set = None
        self.current_model = {}


    def start(self):
        self.handle_menu_choice()


    def handle_menu_choice(self):
        while True:
            Ui.show_menu()
            choice = input('>>> ')
            match choice:
                case '1':
                   self.train_new_model()

                case '2':
                    new_data = self.get_data_from_user()
                    self.send_new_data_for_classifying(new_data)
                case '3':
                    break
                case _:
                    print('Invalid choice.\n')

    def train_new_model(self):
        data_set = GetData.get_raw_df(self.get_path())
        self.current_data_set = data_set
        training_df, testing_df = self.split_the_data_set(data_set)
        model = self.model_builder.build_model(training_df)
        self.current_model = model
        self.check_model_accuracy(model, testing_df)

    @staticmethod
    def get_path():
        path = input('Enter file path: ')
        return path

    @staticmethod
    def split_the_data_set(df):
        split_index = int(len(df) * 0.70)
        training_df = df.iloc[:split_index]
        checking_df = df.iloc[split_index:]
        return training_df, checking_df

    def send_new_data_for_classifying(self, new_data: dict):
        Classifier.classify_example(new_data, self.current_model)


    @staticmethod
    def check_model_accuracy(model, test_df):
        correct = 0
        total = len(test_df)
        label_col = test_df.columns[-1]  # assuming last column is the label

        for _, row in test_df.iterrows():
            features = row.drop(label_col).to_dict()
            actual = row[label_col]
            predicted = Classifier.classify_example(features, model)
            if predicted == actual:
                correct += 1

        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy


    def get_data_from_user(self):
        df = self.current_data_set
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
                            user_data[col] = unique_vals[choice - 1]
                            break
                        else:
                            print("Invalid choice. Please try again.")
                    except ValueError:
                        print("Please enter a valid number.")
            print("\nYou entered:", user_data)
            return user_data
        except ValueError as e:
            print('An error accord', e)