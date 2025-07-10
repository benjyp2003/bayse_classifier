class NaiveBayesModel:
    def __init__(self, priors, likelihoods):
        self.priors = priors
        self.likelihoods = likelihoods 