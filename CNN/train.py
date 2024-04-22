import numpy as np
import tensorflow as tf

import TF_1D_CNN as yolo

NUM_INPUTS = 20000
NUM_CHANNELS = 1

test_input = tf.random.uniform((NUM_INPUTS, NUM_CHANNELS), dtype=tf.dtypes.float32)

model = yolo.Create_Yolov3(NUM_INPUTS, NUM_CHANNELS, training=False)

# TODO: finish testing this
pred = model(test_input)

