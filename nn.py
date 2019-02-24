import tensorflow

from tensorflow.python import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.optimizers import *
from utils import *
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.layers import Lambda,Dense,Dropout,Activation,Flatten,Input,Conv2D,AveragePooling2D,MaxPooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D,BatchNormalization
from tensorflow.python.keras.models import Model
import os
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.keras.callbacks import LearningRateScheduler
# from cifar10 import *


class M(type):
    @classmethod
    def __prepare__(metacls, name, bases, **kwds):
        d = dict()
        for base in bases:
            for key, value in base.__dict__.items():
                if not key.startswith('_'):
                    d[key] = value
        return d

    def __new__(cls, name, bases, namespace, **kwds):
        for base in bases:
            for key, value in base.__dict__.items():
                if not key.startswith('_'):
                    del namespace[key]
        return type.__new__(cls, name, bases, dict(namespace))

class BaseNN(object):
    trainable = True
    model = None
    sess = None
    input = {}
    output = {}
    var = {}
    op = {}
    config = {}
    framework = 'keras'

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

    # impl net & train_op
    def build_model(self, build_func):
        # 注意python是传对象的引用，因此可以修改self的成员
        self.model = build_func()
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

    def train(self, feed_dict):
        if self.framework == 'keras':
            assert self.model is not None

def lr_schedule(epoch):

    """Learning Rate Schedule



    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.

    Called automatically every epoch as part of callbacks during training.



    # Arguments

        epoch (int): The number of epochs



    # Returns

        lr (float32): learning rate

    """

    lr = 1e-3

    if epoch > 180:

        lr *= 0.5e-3

    elif epoch > 160:

        lr *= 1e-3

    elif epoch > 120:

        lr *= 1e-2

    elif epoch > 80:

        lr *= 1e-1

    print('Learning rate: Learning rate: ', lr)

    # lr_str = os.getenv('keras_lr')
    # if lr_str is not None:
    #     print('lr change to {}'.format(lr))
    #     lr = float(lr_str)
    # else:
    #     print('not get env keras_lr, skip ... ')

    with open('./keras_lr.txt', 'r') as f:
        lr_str = f.read()
        print('lr_str ====== {}'.format(lr_str))
        lr = float(lr_str)
        print('lr change to {}'.format(lr))

    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

class MyCallBack(keras.callbacks.Callback):
    def __init__(self, val_set=[], nn=None, filepath ='', period=1, strategy='best'):
        super(MyCallBack, self).__init__()
        if len(val_set) != 2:
            return
        if nn is None:
            return
        if filepath == '':
            return
        self.nn = nn
        self.val_set = val_set
        self.filepath = filepath
        self.max_val_acc = 0.0
        self.period = period
        # self.epoch = 0
        self.strategy = strategy

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.period != 0:
            return

        # print('\nlog ==== ' + str(logs))

        print('\nnow evaluate using val set in epoch: {}...'.format(epoch))
        val_loss, val_acc = self.nn.evaluate(self.val_set[0], self.val_set[1], verbose=0)

        filepath = self.filepath

        format_dict = {}
        if 'val_acc' in filepath:
            format_dict['val_acc'] = val_acc
        if 'epoch' in filepath:
            format_dict['epoch'] = epoch
        if format_dict:
            filepath = filepath.format(**format_dict)

        if self.strategy == 'save_if_improve':
            if self.max_val_acc < val_acc:
                print('improve val_acc {} ==> {}'.format(self.max_val_acc, val_acc))
                self.max_val_acc = val_acc
                print('store to {}'.format(filepath))
                self.nn.store(save_file_name=filepath)
            else:
                print('no improve val_acc {} vs {}'.format(self.max_val_acc, val_acc))

        if self.strategy == 'save_every_time':
            print('store to {}'.format(filepath))
            self.nn.store(save_file_name=filepath)




# class readonly_NN(BaseNN):
#     def __init__(self):
#         super(BaseNN, self).__init__()
#         self.trainable = False
#
#     def infer(self, _input):
#         pass


class keras_nn(BaseNN):
    def __init__(self):
        super(keras_nn, self).__init__()

        self.config = {
            'keras_callbacks':[],
        }

    # impl net & train_op
    def build_model(self, build_func):
        self.model = build_func()
        self.trainable = True

    def store(self, save_file_name):
        assert self.model is not None
        self.model.save(save_file_name)

    def restore(self, load_file_path):
        assert self.model is None
        self.model = keras.models.load_model(load_file_path)

    def summary(self):
        assert self.model is not None
        self.model.summary()

    def add_callback(self, callback_func):
        if isinstance(callback_func, keras.callbacks.Callback):
            self.config['keras_callbacks'].append(callback_func)
        else:
            print('invalid call back')

    def train(self, x, y, epochs, datagen = None, **kwargs):
        assert self.model is not None
        assert self.trainable
        assert isinstance(self.model, keras.models.training.Model)

        batch_size = int(kwargs['batch_size']) \
            if 'batch_size' in kwargs.keys() else 32
        shuffle = bool(kwargs['shuffle']) \
            if 'shuffle' in kwargs.keys() else True

        callbacks = list(filter(lambda item: isinstance(item, keras.callbacks.Callback), kwargs['callbacks'])) \
            if 'callbacks' in kwargs.keys() and isinstance(kwargs['callbacks'], list) else self.config['keras_callbacks']

        if 'batch_size' in kwargs.keys():
            kwargs.pop('batch_size')
        if 'shuffle' in kwargs.keys():
            kwargs.pop('shuffle')
        if 'callbacks' in kwargs.keys():
            kwargs.pop('callbacks')

        # 如果使用图像增强
        if isinstance(datagen, keras.preprocessing.image.ImageDataGenerator):
            datagen.fit(x)
            self.model.fit_generator(datagen.flow(x, y, batch_size=batch_size),
                                     steps_per_epoch=x.shape[0] // batch_size,
                                     epochs=epochs,
                                     shuffle=shuffle,
                                     callbacks=callbacks)
        else:
            # print('kwargs =' + str(kwargs) )
            self.model.fit(x, y,
                           batch_size=batch_size,
                           # steps_per_epoch=x.shape[0] // batch_size,
                           epochs=epochs,
                           # shuffle=shuffle,
                           callbacks=callbacks,
                           **kwargs
                           )

    def evaluate(self, x, y, **kwargs):
        assert self.model is not None
        return self.model.evaluate(x, y, **kwargs)

    def infer(self, x):
        return self.model.predict(x)


def build_pretrain_modle_resnet50(num_classes = 10, freeze_layers = 2):
    # FREEZE_LAYERS = 2

    base_model = ResNet50(weights='imagenet', include_top=False, classes=num_classes)

    x = base_model.output
    # x = Flatten()(x)
    # x = Dropout(0.5)(x)
    # pred = Dense(10, activation='softmax')(x)
    #
    # model = Model(inputs=base_model.input, outputs=pred)

    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    pred = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=pred)

    for layer in model.layers[:freeze_layers]:
        layer.trainable = False

    for layer in model.layers[freeze_layers:]:
        layer.trainable = True

    return model


    # model = keras.Sequential()
    # model.add(
    #     ResNet50(include_top=False, weights='imagenet',input_tensor=None)
    # )
    # model.add(Flatten())
    # model.add(Dropout(0.5))
    # # model.add(Dense())
    # model.add(Dense(num_classes))
    # model.add(Activation('softmax'))


def build_pretrain_model_mobilenetv2(num_classes = 10, freeze_layers = 2):
    # FREEZE_LAYERS = 2

    # base_model = ResNet50(weights='imagenet', include_top=False, classes=num_classes)
    base_model = MobileNetV2(weights='imagenet', include_top=False, classes=num_classes)

    x = base_model.output
    # x = Flatten()(x)
    # x = Dropout(0.5)(x)
    # pred = Dense(10, activation='softmax')(x)
    #
    # model = Model(inputs=base_model.input, outputs=pred)

    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    pred = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=pred)

    for layer in model.layers[:freeze_layers]:
        layer.trainable = False

    for layer in model.layers[freeze_layers:]:
        layer.trainable = True

    return model



def test_Keras_nn():

    # 数据初始化
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    num_classes = 10

    # model_name = 'd:/model/cifar10_FMP.h5'


    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = (x_train.astype('float32') - 127.5) / 127.5
    x_test = (x_test.astype('float32') - 127.5) / 127.5

    # x_train = x_train[:128,:,:,:]
    # y_train = y_train[:128]
    #
    # x_test = x_train[:64, :, :, :]
    # y_test = y_train[:64]

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # # tbCallBack = keras.callbacks.TensorBoard(log_dir='d:/graph', histogram_freq=0, write_graph=True,
    #                                          write_images=True)
    # # 这该方式不适用与 fit_generator，需要自己实现callback
    # ckptCallBack = keras.callbacks.ModelCheckpoint\
    #     ('d:/model2/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    # # 这该方式不适用与 fit_generator，需要自己实现callback
    # bestSaveCallBack = keras.callbacks.ModelCheckpoint \
    #     ("d:/model2/weights.best.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    #
    # # print_log_callback = keras.callbacks.LambdaCallback(
    # #     on_epoch_end=lambda epoch, logs: print(logs)
    # # )
    #
    # # k.add_callback(tbCallBack)

    k = keras_nn()

    best_model_path = 'd:/mobilev2_weight.best.hdf5'
    # k.build_model(myModel)
    if os.path.exists(best_model_path):
        k.restore(load_file_path=best_model_path)
    else:
        k.build_model(build_pretrain_model_mobilenetv2)


    k.add_callback(
        MyCallBack(
            val_set=(x_test,y_test),
            nn=k,
            filepath=best_model_path,
            period=2,
            strategy='save_if_improve',
        )
    )

    # k.add_callback(
    #     MyCallBack(
    #         val_set=(x_test, y_test),
    #         nn=k,
    #         filepath='d:/model2/zjz_weight.{epoch:02d}.{val_acc:.2f}.hdf5',
    #         period=2,
    #         strategy='save_every_time',
    #     )
    # )

    k.add_callback(
        keras.callbacks.TensorBoard(
            log_dir='d:/graph',
            histogram_freq=0,
            write_graph=True,
            write_images=True)
    )

    k.add_callback(
        lr_scheduler
    )

    k.summary()
    model_json = k.model.to_json()
    with open(r'd:/modle.json', 'w') as file:
        file.write(model_json)
    plot_model(k.model, to_file='d:/modle.png', show_shapes=True, show_layer_names=True)


    k.model.compile(optimizer=SGD(1e-5), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # k.train(x_train, y_train, epochs= 1000, batch_size= 32, datagen=datagen, validation_split = 0.1)

    # k.train(x_train, y_train, epochs=1000, batch_size=32, datagen=None, validation_split=0.1)



    # k.model.fit(x_train, y_train, epochs = 100, batch_size=32, validation_split=0.1, callbacks=[bestSaveCallBack])
    # k.train(x_train, y_train, datagen=datagen, epochs = 100, validation_data = (x_test, y_test))

    k.train(x_train, y_train, datagen=datagen, epochs=100)

    # pass


if __name__ == '__main__':
    test_Keras_nn()
    pass









