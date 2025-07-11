import requests
from controller.classifying_manager import ClassifyingManager
from controller.training_manager import TrainingManager
from ui.cli import CLI
from core.trainer import Trainer


class Manager:
    def __init__(self):
        """Initialize the Manager with core builder, current dataset, and current core."""
        self.training_manager = TrainingManager()
        self.classifying_manager = ClassifyingManager()

        self.__models = []
        self.model_builder = Trainer()
        self.current_data_set = None
        self.current_model = {}


    def start(self):
        """Start the main menu loop."""
        self.handle_menu_choice()


    def handle_menu_choice(self):
        """Display menu and handle user choices for training, classifying, or exiting."""
        while True:
            CLI.show_menu()
            choice = input('>>> ')
            match choice:
                case '1':
                    self.training_manager.process_new_model()
                case '2':
                    new_data = self.classifying_manager.get_data_from_user_for_classifying()
                    self.classifying_manager.send_new_data_for_classifying(new_data)
                case '3':
                    break
                case _:
                    print('Invalid choice.\n')

