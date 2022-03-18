'''
Convert excel entries into "tfrecord" file type for Vitis-AI Quantizer
https://github.com/Xilinx/Vitis-AI-Tutorials/tree/master/Design_Tutorials/08-tf2_flow/files
'''
import numpy as np
import tensorflow as tf
import csv

def tfr_parse(np_arr, label):

    data = {
        'height': _int64_feature(1),
        'width': _int64_feature(9),
        'depth': _int64_feature(1),
        'raw_bytes': _float_feature(np_arr),
        'label': _int64_feature(label)
    }

    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out

def write_images_to_tfr_short(images, labels, filename:str="data"):
    filename= filename+".tfrecords"
    writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
    count = 0

    dataset = "dataset/quantize/test.csv"


    for index in range(len(images)):

    #get the data we want to write
    current_image = images[index] 
    current_label = labels[index]

    out = tfr_parse(image=current_image, label=current_label)
    writer.write(out.SerializeToString())
    count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count