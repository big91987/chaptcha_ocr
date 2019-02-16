import tensorflow

from tensorflow.python import keras
from utils import *

# class BaseNN(object):
#
#     def __init__(self):
#         self.model_source = ''
#         self.trainable = True
#         self.model = None
#
#     # 从头建立 model
#     def build_model(self):
#         assert self.model_source == ''
#         self.model_source = 'build'
#         self.trainable = True
#         pass
#
#     # 从pb读取 model， 仅能推理
#     def import_pb(self, filename):
#         assert self.model_source == ''
#         self.trainable = False
#         self.model_source = 'pb'
#         pass
#
#     def export_pb(self, filename):
#         assert self.model is not None
#         assert self.model_source != ''
#         pass
#
#     def import_ckpt(self, path):
#         assert self.model_source == ''
#         self.trainable = True
#         self.model_source = 'ckpt'
#         pass
#
#     def export_ckpt(self, filename):
#         assert self.model is not None
#         assert self.model_source != ''
#         pass
#
#     def generate_log(self):
#         pass
#
#     def train(self, feed_dict):
#         assert self.model_source != ''
#         assert self.model is not None
#         assert self.trainable is True
#         pass
#
#     def infer(self, _input):
#         assert self.model_source != ''
#         assert self.model is not None
#         pass


# class resnet50(BaseNN):
#     def __init__(self):
#         super(BaseNN, self).__init__()
#
#     def build_model(self):








# class vgg16(BaseNN):
#     def __init__(self):
#         super(BaseNN,self).__init__()
#         build_model()
#         pass
#
#     def build_model(self):
#         self.model = keras.Sequential
#
#
#
#
# def build_model(
#         img_shape=(60, 160, 3),
#         n_char=4,
#         batch_size=64,
#         char_set=char_set,
#         char_bag=char_bag):
#     # input =
#
#     if len(img_shape) == 2:
#         input_shape = [None, img_shape[0], img_shape[1], 1]
#     else:
#         input_shape = [None, img_shape[0], img_shape[1], img_shape[2]]
#
#     model = keras.Sequential()
#     model.add(
#         keras.layers.Conv2D(
#             input_shape=input_shape,
#             filters=5,
#             kernel_size=(3, 3),
#             strides=(1, 1),
#             activation=tf.nn.relu)
#     )
#     model.add(
#         keras.layers.Conv2D(
#             # input_shape=input_shape,
#             filters=5,
#             kernel_size=(3, 3),
#             strides=(1, 1),
#             activation=tf.nn.relu)
#     )
#     model.add(
#         keras.layers.Flatten()
#     )
#     model.add(
#         keras.layers.Dense(
#             len(char_bag) * n_char
#         )
#     )
#
#     pass
#
#     # print(tmp)



class BaseNN(object):

    def __init__(self):
        self.trainable = True
        self.model     = None
        self.sess      = None
        self.input     = {}
        self.output    = {}
        self.var       = {}
        self.op        = {}
        self.config    = {}
        self.framework = 'keras'

    # 从头建立 model
    def build_model(self, build_func):
        raise NotImplementedError('not impl yet')

    # 从pb读取 model， 仅能推理
    def import_pb(self, filename):
        raise NotImplementedError('not impl yet')

    def load_pb(self, filename):
        raise NotImplementedError('not impl yet')

    def store(self, path):
        raise NotImplementedError('not impl yet')

    def restore(self, filename):
        raise NotImplementedError('not impl yet')

    def generate_log(self):
        raise NotImplementedError('not impl yet')

    def train(self, feed_dict):
        raise NotImplementedError('not impl yet')

    def infer(self, _input):
        raise NotImplementedError('not impl yet')

class trainable_NN(BaseNN):
    def __init__(self):
        super(BaseNN,self).__init__()
        # self.trainable = True

    def build_model(self, build_func):
        # 注意python是传对象的引用，因此可以修改self的成员
        self.model = build_func(self)
        self.trainable = True

    def store(self, save_file_name):

        if self.framework == 'keras':
            assert self.model is not None
            self.model.save(save_file_name)
        if self.framework == 'tensorflow':
            assert self.sess is not None
            saver = tf.train.Saver()
            saver.save(self.sess, save_file_name)

    def restore(self, load_file_path):

        if self.framework == 'keras':
            assert self.model is None
            self.model = keras.models.load_model(load_file_path)
        if self.framework == 'tensorflow':
            assert self.sess is not None
            ckpt_state = tf.train.get_checkpoint_state(load_file_path)
            if not ckpt_state:
                print('restore failed! ')
            saver = tf.train.Saver()
            saver.restore(self.sess, ckpt_state.model_checkpoint_path)



class readonly_NN(BaseNN):
    def __init__(self):
        super(BaseNN, self).__init__()
        self.trainable = False

    def infer(self, _input):

