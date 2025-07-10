
class Classifier:

    @staticmethod
    def classify_example(new_example, model):
        """Calculate the posterior probabilities for each class
        This involves multiplying the prior probability by the likelihood of the new example
        given each class."""
        priors = model.priors
        likelihoods = model.likelihoods
        mult_results = {}

        for class_label in likelihoods:
            mult_results[class_label] = priors[class_label]

            for feature, feature_value in new_example.items():
                feature_probs = likelihoods[class_label].get(feature, {})
                prob = feature_probs.get(feature_value, 1)  # default to 1 if unseen (not ideal but keeps it simple)
                mult_results[class_label] *= prob  # multiply P(X|C)

        if mult_results:
            predicted = max(mult_results, key=lambda k: mult_results[k])
            return predicted
        else:
            print("No results to classify.")
            return None

