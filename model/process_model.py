from data.get_raw_data import GetData
from model.NaiveBayesModel import NaiveBayesModel

class ModelProcessing:

    def build_model(self, df):

        features = ModelProcessing.get_features_except_last(df)
        target_class = ModelProcessing.get_target_class(df)

        # Step 1: Calculate prior probabilities P(C)
        class_priors = target_class.value_counts(normalize=True).to_dict()

        # Step 2: Calculate conditional probabilities P(features|C) with Laplace smoothing
        likelihoods = {}

        for class_label in target_class.unique():
            features_mask = features[target_class == class_label]  # Subset where class = class_label
            likelihoods[class_label] = {}

            for col in features.columns:
                feature_values = features[col].unique()  # All possible values of the feature
                value_counts = features_mask[col].value_counts()  # Raw counts (not normalized)
                total_count = value_counts.sum()
                num_unique_values = len(feature_values)

                # Initialize inner dict for this column
                likelihoods[class_label][col] = {}

                for val in feature_values:
                    count = value_counts.get(val, 0)  # Get count or 0 if not found
                    # Laplace smoothing formula:
                    smoothed_prob = (count + 1) / (total_count + num_unique_values)
                    likelihoods[class_label][col][val] = smoothed_prob

        return NaiveBayesModel(class_priors, likelihoods)

    @staticmethod
    def get_target_class(df):
        return df.iloc[:, -1]

    @staticmethod
    def get_features_except_last(df):
        return df.iloc[:, :-1]

    @staticmethod
    def check_accuracy(model, test_data):
        pass

