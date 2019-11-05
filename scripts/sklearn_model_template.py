import numpy as np
import sklearn.neural_network
import pickle
import time

#nn name
nn_name = "test"

# data
x_input = "./data/9_params_21_2_x_phys_tr.csv"
y_input = "./data/9_params_21_2_y_phys_tr.csv"
x_train = np.genfromtxt(x_input, delimiter=",")
y_train = np.genfromtxt(y_input, delimiter=",")

#architecture parameters
num_inputs = x_train.shape[1] #aka n in ml terminology
num_outputs = y_train.shape[1] #dimensionality of output y (for single record)
layer_sizes = [16] #INSERT LIST HERE E.G. [16,16]

#propagation parameters
epochs = 2 #INSERT NUMBER HERE
m = x_train.shape[0] #total number of records
batch_num = int(m / 2) #INSERT FRACTION OF m HERE E.G. int(m / 2)

#other mlp parameters
activation = 'tanh' #INSERT ACTIVATION TYPE HERE E.G. 'tanh'
optimiser = 'adam'

#get sklearn model
model = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=layer_sizes, activation=activation, solver=optimiser, alpha=0.000, batch_size=batch_num, learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=epochs, shuffle=True, random_state=None, tol=0.0001, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0., beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=1000)

#output information about run
print "sklearn model run with following data/parameters:"
print "nn name = "
print nn_name
print "x_input = "
print x_input
print "y_input = "
print y_input
print "layer sizes = "
print layer_sizes
print "number of epochs = "
print epochs
print "m = "
print m
print "batch number = "
print batch_num
print "activation type = "
print activation
print "optimiser type = "
print optimiser

start_time = time.time()

#train model
model.fit(x_train, y_train)

#save model
pickle.dump(model, open("./saved_models/sklearn/" + nn_name + ".sav", 'wb'))

end_time = time.time()

print "model fitting and saving to file took " + str(end_time - start_time) + " seconds or " + str((end_time - start_time) / 3600.) + " hours"
 
