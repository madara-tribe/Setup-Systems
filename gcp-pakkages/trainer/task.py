import tensorflow as tf
from . import train

if __name__=='__main__':
    tf.logging.set_verbosity("INFO")
    train.train()