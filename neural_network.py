import numpy as np
import tensorflow as tf
import pickle

from tensorflow import keras
from tensorflow.keras import layers

def residual_block(x_input, layers_resblock, resblock_layer_size, batch_norm):
    x = x_input
    for i in range(layers_resblock - 1):
        x = layers.Dense(resblock_layer_size)(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.Dense(resblock_layer_size)(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Add()([x_input, x])
    x = layers.ReLU()(x)
    return x

def _nn_model(name, input_shape=324, first_layers_neurons=(5000,1000), n_resblocks=4, layers_resblock=2, batch_norm=True):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for n in first_layers_neurons:
        x = layers.Dense(n)(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    
    for i in range(n_resblocks):
        x = residual_block(x, layers_resblock, first_layers_neurons[-1], batch_norm)
        
    output = layers.Dense(1)(x)
    
    model = keras.Model(inputs, output, name=name)
    return model

def nn_deepcubea(input_shape=324):
    return _nn_model("DNN_0", input_shape=input_shape)

def nn_deepcubea_reduced0(input_shape=324):
    return _nn_model("DNN_reduced_0", input_shape=input_shape, first_layers_neurons=(3000,800), n_resblocks=3, layers_resblock=2)

def nn_deepcubea_noBatchNorm(input_shape=324):
    # B=1000, B_val=200, K=20, M=10000, C=10
    return _nn_model("DNN_noBatchNorm", input_shape=input_shape, batch_norm=False)
    
def nn_deepcubea_final_2x2x2(input_shape=6*4*6):
    return _nn_model("DNN_final_2x2x2", input_shape=input_shape, first_layers_neurons=(3000,800), n_resblocks=3, layers_resblock=2)
    
def nn_deepcubea_final_2x2x2_nr(input_shape=6*4*6):
    return _nn_model("DNN_final_2x2x2_nr", input_shape=input_shape, first_layers_neurons=(3000,800), n_resblocks=3, layers_resblock=2)

def nn_deepcubea_final_2x2x2_nrr(input_shape=6*4*6):
    return _nn_model("DNN_final_2x2x2_nrr", input_shape=input_shape, first_layers_neurons=(3000,800), n_resblocks=3, layers_resblock=2)

def nn_deepcubea_reduced0_noBatchNorm(input_shape=324):
    return _nn_model("DNN_reduced_0_noBatchNorm", input_shape=input_shape, first_layers_neurons=(3000,800), n_resblocks=3, layers_resblock=2, batch_norm=False)
    
def model_factory(model_name, input_shape):
    if model_name == "DNN_0":
        return nn_deepcubea(input_shape)
    elif model_name == "DNN_reduced_0":
        return nn_deepcubea_reduced0(input_shape)
    elif model_name == "DNN_noBatchNorm":
        return nn_deepcubea_noBatchNorm(input_shape)
    elif model_name == "DNN_reduced_0_noBatchNorm":
        return nn_deepcubea_reduced0_noBatchNorm(input_shape)
    elif model_name == "DNN_final_2x2x2":
        return nn_deepcubea_final_2x2x2(input_shape) # eps = 0.07
    elif model_name == "DNN_final_2x2x2_nr":
        return nn_deepcubea_final_2x2x2_nr(input_shape) # does not train wrong values
    elif model_name == "DNN_final_2x2x2_nrr":
        return nn_deepcubea_final_2x2x2_nrr(input_shape) # does not train wrong values
    
def load_weights(model, puzzle_name, iteration, save_path):
    return model.load_weights(save_path + "model_e/" + puzzle_name + "/" + model.name + "_" + str(iteration))

def save_weights(model, puzzle_name, iteration, save_path):
    model.save_weights(save_path + "model_e/" + puzzle_name + "/" + model.name + "_" + str(iteration))
    
def load_losses(model_name, puzzle_name, iteration, save_path):
    with open(save_path + "losses/"  + puzzle_name + "/" + model_name + "_" + str(iteration), 'rb') as f:
        losses = pickle.load(f)
    return losses

def save_losses(model_name, puzzle_name, iteration, save_path, losses):
    with open(save_path + "losses/" + puzzle_name + "/" + model_name + "_" + str(iteration), 'wb+') as f:
        pickle.dump(losses, f)
        
def load_heuristic(model_name, puzzle_name, iteration, save_path, input_shape=6*4*6):
    model = model_factory(model_name, input_shape=input_shape)
    nn.load_weights(model, puzzle_name, iteration, save_path).expect_partial()
    return model
