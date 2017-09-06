from feature_format import train, test 
import numpy as np 

# Sklearn
from sklearn.mixture import BayesianGaussianMixture 
model = BayesianGaussianMixture(n_components = 4)
model.fit(train)

print model.predict(test)

# My model
from gmm import GaussianMixture 

my_model = GaussianMixture(K = 4)
my_model.fit(train)

# print my_model.predict(test)