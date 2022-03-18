import argparse
import os
import shutil
import sys

import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

def load_data(data_dir):


    return dataset

def quant_model(float_model, quant_model, batchsize, tfrec_dir, evaluate): # FP32 -> INT8
    # make folder for saving quantized model
    head_tail = os.path.split(quant_model) 
    os.makedirs(head_tail[0], exist_ok = True)

    # load the floating point trained model
    float_model = load_model(float_model)

    # get input dimensions of the floating-point model
    height = float_model.input_shape[1]
    width = float_model.input_shape[2]

    # load / pre-process any data
    # Needs to be a "TFRecord"?
    quant_dataset = 

    # Quantize model and save
    quantizer = vitis_quantize.VitisQuantizer(float_model)
    quantized_model = quantizer.quantize_model(calib_dataset=quant_dataset)
    quantized_model.save(quant_model)

    print("QUANTIZED MODEL SAVED TO: ", quant_model)