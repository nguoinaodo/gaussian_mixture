from feature_format import train, test 
import numpy as np 

# Sklearn
from sklearn.mixture import BayesianGaussianMixture 
model = BayesianGaussianMixture(n_components = 5)
model.fit(train)

print 'test length: %d' % len(test) 
pred1 = model.predict(test)
print pred1
# My model
print "My model"
from gmm import GaussianMixture 

my_model = GaussianMixture(K = 5)
my_model.fit(train)

pred2 = my_model.predict(test)
print pred2
print np.where(pred1 == pred2)[0].shape
