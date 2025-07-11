

class Trainer:

    def build_model(self, df, model_name):
        """
        Builds a Naive Bayes core from the input DataFrame or list of dicts.
        This function calculates prior probabilities and conditional probabilities
        for each class and feature, using Laplace smoothing.
        """
        import pandas as pd
        # Accept both DataFrame and list of dicts
        if isinstance(df, list):
            df = pd.DataFrame(df)

        features = Trainer.get_features_except_last(df)
        target_class = Trainer.get_target_class(df)

        # Step 1: Calculate prior probabilities P(C)
        class_priors = target_class.value_counts(normalize=True).to_dict()

        # Step 2: Calculate conditional probabilities P(features|C) with Laplace smoothing
        likelihoods = {}

        for class_label in target_class.unique():
            features_mask = features[target_class == class_label]  # Subset where class = class_label
            class_label_str = str(class_label)
            likelihoods[class_label_str] = {}

            for col in features.columns:
                feature_values = features[col].unique()  # All possible values of the feature
                value_counts = features_mask[col].value_counts()  # Raw counts (not normalized)
                total_count = value_counts.sum()
                num_unique_values = len(feature_values)

                # Initialize inner dict for this column
                likelihoods[class_label_str][col] = {}

                for val in feature_values:
                    str_val = str(val)
                    count = value_counts.get(val, 0)  # Get count or 0 if not found
                    # Laplace smoothing formula:
                    smoothed_prob = (count + 1) / (total_count + num_unique_values)
                    likelihoods[class_label_str][col][str_val] = smoothed_prob

        return {model_name: {'priors': class_priors, 'likelihoods': likelihoods}}

    @staticmethod
    def get_target_class(df):
        """
        Extracts the target class column from the DataFrame.
        The target class is assumed to be the last column.
        """
        return df.iloc[:, -1]

    @staticmethod
    def get_features_except_last(df):
        """
        Extracts all feature columns from the DataFrame, excluding the target class.
        """
        return df.iloc[:, :-1]


