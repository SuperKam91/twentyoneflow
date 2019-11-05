#import modules
import numpy as np
import tensorflow as tf
import time
import os

#import in-house software
import keras_losses as kl
import keras_tools as kt
import keras_models as km
import metrics as mets

#nn name
nn_name = "8_params_phys_mlp_64_0"

# data
all_params = False
if all_params:
    par_slice = range(9)
else:
    par_slice = range(7) + range(8,9)

# training data
x_train = np.genfromtxt("./data/9_params_21_2_x_phys_tr.csv", delimiter=",")[:,par_slice]
y_train = np.genfromtxt("./data/9_params_21_2_y_phys_tr.csv", delimiter=",")
y_train = y_train.reshape(-1,1) #keras outputs y_pred as 2d, so make y_true 2d as well

#architecture parameters
num_inputs = x_train.shape[1] #aka n in ml terminology
num_outputs = y_train.shape[1] #dimensionality of output y (for single record)
layer_sizes = [64] #INSERT LIST HERE E.G. [16,16]

#propagation parameters
epochs = 20000 #INSERT NUMBER HERE
m = x_train.shape[0] #total number of records
batch_num = m #INSERT FRACTION OF m HERE E.G. int(m / 2)

#other nn hyperparameters
dropout_reg = 0.0

#get keras model
activation = 'tanh' #INSERT ACTIVATION TYPE HERE E.G. 'tanh'
model = km.mlp(num_inputs, num_outputs, layer_sizes, activation) #INSERT KERAS MODEL HERE FROM km. E.G. mlp(num_inputs, num_outputs, layer_sizes, activation)

#following are required for certain losses/metrics
mean = -49.72584314
var = 5524.38030468
n_z = 136

#specify keras model loss function to optimise e.g.
losses = ['mean_squared_error']

#specify keras model metric e.g.
metrics = [kl.twenty_one_cm_rmse_higher_order(mean, var)] #rmse over data not ts

#hyperparameters for some optimisers e.g. adam
lr = 0.001
beta_1 = 0.9
beta_2 = 0.999

#choose optimiser
optimiser = tf.keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=None, decay=0.0, amsgrad=False)

#compile model
model.compile(optimizer=optimiser, loss=losses, metrics=metrics)

#starting point for model optimisation
initial_epoch = 0

#create model callback to save model after every n_save epochs
save_file = "./saved_models/keras/" + nn_name + ".h5"
n_save = 5000
save_callback = tf.keras.callbacks.ModelCheckpoint(save_file, period = n_save)

#check if model already exists
if os.path.isfile(save_file):
	#if so, load in saved model state and resume training on that
	print "saved model found. loading in and resuming training on this model"
	print "note, training will resume from last (saved) epoch found in output file"
	initial_epoch = kt.get_epoch_from_output(nn_name, initial_epoch, epochs, n_save)
else:
	#output information about run
	print "keras model run with following data/parameters:"
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
	print "dropout reg = "
	print dropout_reg
	print "activation type = "
	print activation
	print "model type = "
	print model
	print "mean of unscaled ouput = "
	print mean
	print "variance of unscaled output = "
	print var
	print "number of points per timeseries (number of z bins) = "
	print n_z
	print "loss functions = "
	print losses
	print "metric functions = "
	print metrics
	print "optimiser = "
	print optimiser
	print "optimiser lr = "
	print lr
	print "optimiser beta_1 = "
	print beta_1
	print "optimiser beta_2 = "
	print beta_2
	print "model summary = "
	print model.summary()
	print "starting epoch = "
	print initial_epoch
	print ""

start_time = time.time()

#train model
history = model.fit(x_train, y_train, batch_size = batch_num, epochs = epochs, initial_epoch = initial_epoch, verbose = 1, callbacks = [save_callback])

#save model in case it wasn't saved during .fit()
model.save(save_file)

end_time = time.time()

print "model fitting and saving to file took " + str(end_time - start_time) + " seconds or " + str((end_time - start_time) / 3600.) + " hours"
