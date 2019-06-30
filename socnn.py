import keras
from keras.layers.convolutional import Conv1D
from keras.layers import Input, Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Permute
from keras.constraints import nonneg, maxnorm
import yaml

with open('config.yml', 'r') as yml:
    config = yaml.load(yml)

    num_layer_sig = config['num_layer_sig']
    #num_layer_off = config['num_layer_off']
    norm = config['norm']
    clipnorm = config['clipnorm']
    aux_weight = config['aux_weight']
    lr = config['lr']
    output_length = config['output_length']
    nonnegative = config['nonnegative']
    ks = config['ks']

def build_socnn(input_shape_sig=(128, 1), input_shape_off=(128, 1), dim=1):
    #significant_network
    Input_sig = Input(shape=input_shape_sig, dtype='float32', name='input_sig')
    name = "Significance_Conv_0"
    x = Conv1D(filters=8,kernel_size=ks, padding='same',
           activation='linear', name=name,
           kernel_constraint=maxnorm(norm))(Input_sig)

    for i in range(num_layer_sig-1):
        name = "Significance_Conv_" + str(i+1)
        if i == (num_layer_sig-2):
            fn = dim-1
        else:
            fn = 8
        x = Conv1D(filters=fn,
                kernel_size=ks, padding='same',
                activation='linear', name=name,
                kernel_constraint=maxnorm(norm))(x)

        x = BatchNormalization(name="Significance_BN"+str(i+1))(x)
    output_sig = x

    #offset_network
    Input_off = Input(shape=input_shape_off, dtype='float32', name='input_off')
    name = "Offset_Conv_0"
    y = Conv1D(filters=dim-1,
            kernel_size=ks, padding='same',
            activation='linear', name=name,
            kernel_constraint=maxnorm(norm))(Input_off)

    output_off = keras.layers.add([y, Input_off], name='output_off')
    value = Permute((2, 1))(output_off)

    output_sig = Permute((2, 1))(output_sig)
    output_sig = TimeDistributed(Activation('softmax'), name='softmax')(output_sig)

    #Hn-1 =  𝝈(𝑺) ⨂(𝐨𝐟𝐟+𝒙𝑰)
    H1 = keras.layers.multiply(inputs=[output_sig, value], name='significancemerge')
    #Hn
    H2 = TimeDistributed(Dense(output_length, activation='linear', use_bias=False,
                                kernel_constraint=nonneg() if nonnegative else None),
                                name='out')(H1)
    main_output = Permute((2, 1), name='main_output')(H2)

    model = keras.models.Model(inputs=[Input_sig, Input_off], outputs=[main_output, output_off])
    model.compile(optimizer=keras.optimizers.Adam(lr=lr, clipnorm=clipnorm),
               loss={'main_output': 'mse', 'output_off': 'mse'},
               loss_weights={'main_output': 1., 'output_off': aux_weight})

    return model


