#import keras
import tensorflow as tf
from tensorflow.python import keras

from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.layers import Lambda,Dense,Dropout,Activation,Flatten,Input,Conv2D,AveragePooling2D,MaxPooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D,BatchNormalization
from tensorflow.python.keras import layers
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import regularizers
import pickle
from lsuv_init import LSUVinit
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
# from tensorflow.nn import fractional_max_pool
# from tensorflow.nn import fractional_max_pool

def identity_block(input_tensor,kernel_size,filters,stage,block):
    filters1,filters2,filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branck'
    x = Conv2D(filters1,(1,1),name = conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis = bn_axis,name = bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters2,kernel_size,padding = 'same',name = conv_name_base +'2b')(x)
    x = BatchNormalization(axis = bn_axis,name = bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters3,(1,1),name = conv_name_base + '2c')(x)
    x = BatchNormalization(axis = bn_axis,name = bn_name_base + '2c')(x)
    x = layers.add([x,input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor,kernel_size,filters,stage,block,strides = (2,2)):
    filters1,filters2,filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branck'
    bn_name_base = 'bn' + str(stage) + block + '_branck'

    x = Conv2D(filters1,(1,1),strides = strides,name = conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis = bn_axis,name = bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2,kernel_size,padding = 'same',name = conv_name_base + '2b')(x)
    x = BatchNormalization(axis = bn_axis,name = bn_name_base + '2b')(x)
    x = Activation('relu')(x)


    x = Conv2D(filters3,(1,1),name = conv_name_base + '2c')(x)
    x = BatchNormalization(axis = bn_axis,name = bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3,(1,1),strides = strides,name = conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis = bn_axis,name = bn_name_base + '1')(shortcut)

    x = layers.add([x,shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50(include_top = True,weights = 'imagenet',input_tensor = None,input_shape = None,pooling = None,classes = 1000):
    bn_axis = 3
    x = Conv2D(64,(7,7),strides = (2,2),padding = 'same',name = 'conv1')(input_tensor)
    x = BatchNormalization(axis = bn_axis,name = 'bn_conv1')(x)
    x = MaxPooling2D((3,3),strides = (2,2))(x)

    x = conv_block(x,3,[64,64,256],stage = 2,block = 'a',strides = (1,1))
    x = identity_block(x,3,[64,64,256],stage = 2,block = 'b')
    x = identity_block(x,3,[64,64,256],stage = 2,block = 'c')

    x = conv_block(x,3,[128,128,512],stage = 3,block = 'a')
    x = identity_block(x,3,[128,128,512],stage = 3,block = 'b')
    x = identity_block(x,3,[128,128,512],stage = 3,block = 'c')
    x = identity_block(x,3,[128,128,512],stage = 3,block = 'd')

    x = conv_block(x,3,[256,256,1024],stage = 4,block = 'a')
    x = identity_block(x,3,[256,256,1024],stage = 4,block = 'b')
    x = identity_block(x,3,[256,256,1024],stage = 4,block = 'c')
    x = identity_block(x,3,[256,256,1024],stage = 4,block = 'd')
    x = identity_block(x,3,[256,256,1024],stage = 4,block = 'e')
    x = identity_block(x,3,[256,256,1024],stage = 4,block = 'f')

    x = conv_block(x,3,[512,512,2048],stage = 5,block = 'a')
    x = identity_block(x,3,[512,512,2048],stage = 5,block = 'b')
    x = identity_block(x,3,[512,512,2048],stage = 5,block = 'c')

    x = AveragePooling2D((1,1),name = 'avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes,activation = 'softmax',name = 'fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
    model = Model(input_tensor,x,name = 'resnet50')
    return model



def VGG16(include_top = True,weights = None,input_tensor = None,input_shape = None,pooling = None,classes = 1000):
    x = Conv2D(64,(3,3),activation = 'relu',padding = 'same',name = 'block1_conv1')(input_tensor)
    x = Conv2D(64,(3,3),activation = 'relu',padding = 'same',name = 'block1_conv2')(x)
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block1_pool')(x)

    #Block2
    x = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv1')(x)
    x = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv2')(x)
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block2_pool')(x)

    #Block3
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv1')(x)
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv2')(x)
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv3')(x)
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block3_pool')(x)

    #Block4
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same',name = 'block4_conv1')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same',name = 'block4_conv2')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same',name = 'block4_conv3')(x)
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block4_pool')(x)

    #Block5
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same',name = 'block5_conv1')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same',name = 'block5_conv2')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same',name = 'block5_conv3')(x)
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block5_pool')(x)

    if include_top:
        x = Flatten(name = 'flatten')(x)
        x = Dense(4096,activation = 'relu',name = 'fc1',kernel_regularizer='l2', bias_regularizer='l2', activity_regularizer='l2')(x)
        x = Dense(4096,activation = 'relu',name = 'fc2',kernel_regularizer='l2', bias_regularizer='l2', activity_regularizer='l2')(x)
        x = Dense(classes,activation = 'softmax',name = 'predictions',kernel_regularizer='l2', bias_regularizer='l2', activity_regularizer='l2')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    model = Model(input_tensor,x,name = 'vgg16')
    return model

def frac_max_pool(x):
    p_ratio = [1.0, 1.44, 1.44, 1.0]
    result = tf.nn.fractional_max_pool(x,p_ratio)[0]
    return result

def VGG16_FMP(include_top = True,weights = None,input_tensor = None,input_shape = None,pooling = None,classes = 1000):
    x = Conv2D(64,(3,3),activation = 'relu',padding = 'same',name = 'block1_conv1')(input_tensor)
    x = Conv2D(64,(3,3),activation = 'relu',padding = 'same',name = 'block1_conv2')(x)
    # x = MaxPooling2D((2,2),strides = (2,2),name = 'block1_pool')(x)
    x = Lambda(frac_max_pool,name = 'block1_pool')(x)
    #Block2
    x = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv1')(x)
    x = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv2')(x)
    # x = MaxPooling2D((2,2),strides = (2,2),name = 'block2_pool')(x)
    x = Lambda(frac_max_pool,name = 'block2_pool')(x)
    #Block3
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv1')(x)
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv2')(x)
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv3')(x)
    # x = MaxPooling2D((2,2),strides = (2,2),name = 'block3_pool')(x)
    x = Lambda(frac_max_pool,name = 'block3_pool')(x)
    #Block4
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same',name = 'block4_conv1')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same',name = 'block4_conv2')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same',name = 'block4_conv3')(x)
    # x = MaxPooling2D((2,2),strides = (2,2),name = 'block4_pool')(x)
    x = Lambda(frac_max_pool,name = 'block4_pool')(x)
    #Block5
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same',name = 'block5_conv1')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same',name = 'block5_conv2')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same',name = 'block5_conv3')(x)
    # x = MaxPooling2D((2,2),strides = (2,2),name = 'block5_pool')(x)
    x = Lambda(frac_max_pool,name = 'block5_pool')(x)
    if include_top:
        x = Flatten(name = 'flatten')(x)
        x = Dense(4096,activation = 'relu',name = 'fc1',kernel_regularizer='l2', bias_regularizer='l2', activity_regularizer='l2')(x)
        x = Dense(4096,activation = 'relu',name = 'fc2',kernel_regularizer='l2', bias_regularizer='l2', activity_regularizer='l2')(x)
        x = Dense(classes,activation = 'softmax',name = 'predictions',kernel_regularizer='l2', bias_regularizer='l2', activity_regularizer='l2')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    model = Model(input_tensor,x,name = 'vgg16')
    return model
#构建不同的模型进行尝试
def myModel():
    input_tensor = Input(shape = (32,32,3))
    base_model = VGG16_FMP(input_tensor = input_tensor,weights = None,include_top = False,pooling = 'avg')
    x = base_model.output
    x = Dropout(0.5)(x)
    x = Dense(512,activation = 'relu',use_bias = False,name = 'fc')(x)
    x = Dense(10,activation= 'softmax',use_bias = False,name = 'cifar10',kernel_regularizer=regularizers.l2(0.0001),  activity_regularizer=regularizers.l1(0.0001))(x)
    model = Model(input_tensor,x)
    return model


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

model_name = 'd:/model/cifar10_FMP.h5'
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train = (x_train.astype('float32') - 127.5)/127.5
x_test = (x_test.astype('float32') - 127.5)/127.5

y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)
a = []
for ii in range(0,30):
    if ii == 0:
        model = myModel()
    else:
        model = load_model(model_name + '_' + str(ii-1))
    model.compile(optimizer = SGD(1e-3),loss = 'categorical_crossentropy',metrics = ['categorical_accuracy'])
    if ii == 0:
        model = LSUVinit(model, x_train[:32, :, :, :])
    tbCallBack = keras.callbacks.TensorBoard(log_dir='D:/Graph', histogram_freq=0, write_graph=True, write_images=True)
    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train,batch_size=32),steps_per_epoch=x_train.shape[0] // 32,epochs = 1,shuffle = True,callbacks=[tbCallBack])
    model.save(model_name+'_'+ str(ii))

    loss,accuracy = model.evaluate(x_test,y_test)
    print('loss: {} accuracy:{}'.format(loss,accuracy))
    a.append(str(accuracy) + '\n')
    with open('d:/model/accurate.txt','w') as fw:
        fw.writelines(a)
