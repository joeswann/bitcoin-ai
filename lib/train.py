import pandas as pd
import tensorflow as tf
from tensorflow.nn.rnn_cell import GRUCell

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

max_length = 100
frame_size = 64
num_hidden = 200

sequence = tf.placeholder(tf.float32, [None, max_length, frame_size])
output, state = tf.nn.dynamic_rnn(
    GRUCell(num_hidden),
    sequence,
    dtype=tf.float32,
    sequence_length=length(sequence),
)