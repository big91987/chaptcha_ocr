import tensorflow

from tensorflow.python import keras
from utils import *

class BaseNN(object):

    def __init__(self):
        self.model_source = ''
        self.trainable = True
        self.model = None

    # 从头建立 model
    def build_model(self):
        assert self.model_source == ''
        self.model_source = 'build'
        self.trainable = True
        pass

    # 从pb读取 model， 仅能推理
    def import_pb(self, filename):
        assert self.model_source == ''
        self.trainable = False
        self.model_source = 'pb'
        pass

    def export_pb(self, filename):
        assert self.model is not None
        assert self.model_source != ''
        pass

    def import_ckpt(self, path):
        assert self.model_source == ''
        self.trainable = True
        self.model_source = 'ckpt'
        pass

    def export_ckpt(self, filename):
        assert self.model is not None
        assert self.model_source != ''
        pass

    def generate_log(self):
        pass

    def train(self, feed_dict):
        assert self.model_source != ''
        assert self.model is not None
        assert self.trainable is True
        pass

    def infer(self, _input):
        assert self.model_source != ''
        assert self.model is not None
        pass


class vgg16(BaseNN):
    def __init__(self):
        super(BaseNN,self).__init__()
        build_model()
        pass

    def build_model(self):
        self.model = keras.Sequential




def build_model(
        img_shape=(60, 160, 3),
        n_char=4,
        batch_size=64,
        char_set=char_set,
        char_bag=char_bag):
    # input =

    if len(img_shape) == 2:
        input_shape = [None, img_shape[0], img_shape[1], 1]
    else:
        input_shape = [None, img_shape[0], img_shape[1], img_shape[2]]

    model = keras.Sequential()
    model.add(
        keras.layers.Conv2D(
            input_shape=input_shape,
            filters=5,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation=tf.nn.relu)
    )
    model.add(
        keras.layers.Conv2D(
            # input_shape=input_shape,
            filters=5,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation=tf.nn.relu)
    )
    model.add(
        keras.layers.Flatten()
    )
    model.add(
        keras.layers.Dense(
            len(char_bag) * n_char
        )
    )

    pass

    # print(tmp)
