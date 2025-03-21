from keras.layers import Input
from tensorflow.keras.layers import concatenate
from keras.layers.core import Lambda
from keras.models import Model
import tensorflow as tf


def make_parallel(model, gpu_count):

    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        stride = tf.concat([shape[:1]//parts, shape[1:]*0], 0)
        start = stride * idx
        size = tf.concat([shape[:1]//parts, shape[1:]], 0) 
        # Split the batch into equal parts 
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:
                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape,
                                  arguments={'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)
                outputs = model(inputs)
                if not isinstance(outputs, list):
                    outputs = [outputs]
                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])
    # Merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(concatenate(outputs, axis=0))
        return Model(inputs=model.inputs, outputs=merged)
