#########commercial modules
import tensorflow as tf 
import numpy as np

#copied from bnn version of same file

def get_gradients(x, y, model, learning_phase = 0):
	"""
	returns gradients of loss function w.r.t nn parameters,
	(both loss and params are specified by keras model),
	for given input values x and output values y.
	taken from github, but think it works correctly.
	"""
	weights = model.trainable_weights # weight tensors
	gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors
	input_tensors = model.inputs + model.sample_weights + model.targets + [tf.keras.backend.learning_phase()] #sample weights are weights to give to each sample, nothing to do with nn model weights.
	#learning_phase() indicates training or testing. 0 denotes training, 1 testing time. makes a difference to values of gradients returned when using batch norm and dropout, but not clear how.
	get_gradients = tf.keras.backend.function(inputs = input_tensors, outputs = gradients)
	inputs = [x, np.ones(len(x)), y, learning_phase]
	grads = get_gradients(inputs)
	return grads

def get_activations(x, model, learning_phase = 0):
	"""
	returns activations (of each layer) of keras model
	for input x.
	adapted from github, but think it works correctly. watch out for 
	value of learning_phase
	"""
	try:
		layers = [layer.output for layer in model.layers] # all layer outputs
		get_activations = tf.keras.backend.function(inputs = [model.input, tf.keras.backend.learning_phase()], outputs = layers) # evaluation function
		layer_activations = get_activations([x, learning_phase]) #1 for testing, will effect dropout, batch normalisation etc
	except tf.errors.InvalidArgumentError:
		layers = [layer.output for layer in model.layers][1:] # all layer outputs except first (input) layer
		get_activations = tf.keras.backend.function(inputs = [model.input, tf.keras.backend.learning_phase()], outputs = layers)
		layer_activations = get_activations([x, learning_phase]) 
	return layer_activations

def get_epoch_from_output(nn_name, initial_epoch, epochs, n_save):
	"""
	get last trained epoch from output file (verbose of model.fit().
	if this is equal to number of epochs, no more training needs to be done.
	otherwise, training should resume from epoch where model was last saved (defined by the period of saving, n_save),
	in which case, initial_epoch should be updated to reflect this.
	returns the appropriate initial_epoch
	""" 
	output_f = open("./run_outputs/" + nn_name + "_output.txt", 'r')
	output_s = output_f.readlines()
	output_f.close()
	epoch_lines = []
	for line in output_s:                                           
		if 'Epoch ' in line:
			epoch_lines.append(line)
	epoch_ns = []
	for l in epoch_lines:
		epoch_ns.append(l.split('/')[0][6:])
	last_epoch = np.max(np.array(epoch_ns, dtype = np.int))
	resume_epoch = (last_epoch // n_save) * n_save
	if last_epoch == epochs:
		print("saved model has been trained up to specified number of epochs (" + str(epochs) + ")")
	else:
		print("saved model has been trained up to epoch " + str(last_epoch))
		print("resuming from last save point which was at epoch " + str(resume_epoch))
	return resume_epoch

def reset_weights(model):
	"""
	re-initialise weight and bias values
	"""
        session = tf.keras.backend.get_session()
	for layer in model.layers: 
		if hasattr(layer, 'kernel_initializer'):
			layer.kernel.initializer.run(session=session)

