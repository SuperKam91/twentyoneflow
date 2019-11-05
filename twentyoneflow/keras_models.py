import tensorflow as tf

Model = tf.keras.models.Model
Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Batch_norm = tf.keras.layers.BatchNormalization

#mlp networks

def mlp_drop_norm(num_inputs, num_outputs, layer_sizes, dropout_reg, activation):
    """
    arbitrary sized mlp with linear output activation, dropout, batch norm, 
    and (same) specified activation function applied to all hidden layers
    """
    a0 = Input(shape = (num_inputs,))
    inputs = a0
    for layer_size in layer_sizes:
        outputs = Dense(layer_size, activation = activation, kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform')(a0)
        outputs = Dropout(dropout_reg)(outputs)
        outputs = Batch_norm()(outputs)
        a0 = outputs
    #don't want dropout and normalisation for output layer. linear activation
    outputs = Dense(num_outputs, activation = 'linear', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform')(a0)
    model = Model(inputs = inputs, outputs = outputs)
    return model

def mlp_drop(num_inputs, num_outputs, layer_sizes, dropout_reg, activation):
    """
    arbitrary sized mlp with linear output activation, dropout, 
    and (same) specified activation function applied to all hidden layers
    """
    a0 = Input(shape = (num_inputs,))
    inputs = a0
    for layer_size in layer_sizes:
        outputs = Dense(layer_size, activation = activation, kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform')(a0)
        outputs = Dropout(dropout_reg)(outputs)
        a0 = outputs
    #don't want dropout and normalisation for output layer. linear activation
    outputs = Dense(num_outputs, activation = 'linear', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform')(a0)
    model = Model(inputs = inputs, outputs = outputs)
    return model

def mlp_norm(num_inputs, num_outputs, layer_sizes, activation):
    """
    arbitrary sized mlp with linear output activation, batch norm, 
    and (same) specified activation function applied to all hidden layers
    """
    a0 = Input(shape = (num_inputs,))
    inputs = a0
    for layer_size in layer_sizes:
        outputs = Dense(layer_size, activation = activation, kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform')(a0)
        outputs = Batch_norm()(outputs)
        a0 = outputs
    #don't want normalisation for output layer. linear activation
    outputs = Dense(num_outputs, activation = 'linear', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform')(a0)
    model = Model(inputs = inputs, outputs = outputs)
    return model

def mlp(num_inputs, num_outputs, layer_sizes, activation):
    """
    arbitrary sized mlp with linear output activation, 
    and (same) specified activation function applied to all hidden layers
    """
    a0 = Input(shape = (num_inputs,))
    inputs = a0
    for layer_size in layer_sizes:
        outputs = Dense(layer_size, activation = activation, kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform')(a0)
        a0 = outputs
    #linear activation for output layer
    outputs = Dense(num_outputs, activation = 'linear', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform')(a0)
    model = Model(inputs = inputs, outputs = outputs)
    return model

#ResNet networks

def uap_mlp_ResNet_block(num_inputs, a0, activation):
    """
    in this architecture each block consists of two layers, the first
    being one neuron wide, the second being num_inputs wide.
    the first layer contains a bias and a ReLu activation,
    the second does not.
    """
    a1 = Dense(1, activation = activation)(a0)
    a2_part = Dense(num_inputs, activation = 'linear', use_bias = False)(a1)
    return tf.keras.layers.Add()([a0, a2_part])

def coursera_mlp_ResNet_block(num_inputs, a0, activation):
    """
    copied from keras_models.py,
    but with option of specifying activation applied to each block
    """
    a1 = Dense(1, activation = activation)(a0)
    z2_part = Dense(num_inputs, activation = 'linear')(a1)
    z2 = tf.keras.layers.Add()([a0, z2_part])
    return tf.keras.layers.Activation(activation)(z2)

def same_mlp_ResNet_block(num_inputs, a0, activation):
    """
    copied from keras_models.py,
    but with option of specifying activation applied to each block
    """
    a1 = Dense(num_inputs, activation = activation)(a0)
    z2_part = Dense(num_inputs, activation = 'linear')(a1)
    z2 = tf.keras.layers.Add()([a0, z2_part])
    return tf.keras.layers.Activation(activation)(z2)

def mlp_ResNet_1(num_inputs, num_outputs, layer_sizes, activation, ResNet_type = 'uap'):
    """
    all layers are size of input of nn, apart from final layer which obvs conforms to output.
    Have to specify activation applied to each block
    """
    num_blocks = 1
    if ResNet_type == 'uap':
        ResNet_block = uap_mlp_ResNet_block
    elif ResNet_type == 'coursera':
        ResNet_block = coursera_mlp_ResNet_block
    elif ResNet_type == 'same':
        ResNet_block = same_mlp_ResNet_block
    else:
        raise NotImplementedError
    a0 = Input(shape = (num_inputs,))
    inputs = a0
    for _ in range(num_blocks):
        a2 = ResNet_block(num_inputs, a0, activation)
        a0 = a2
    outputs = Dense(num_outputs, activation = 'linear')(a0)
    return Model(inputs = inputs, outputs = outputs)
