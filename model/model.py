import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Multiply, ZeroPadding2D, concatenate, Conv2D, Input, Dense, Reshape, BatchNormalization, Activation, LSTM, Bidirectional, Lambda
from Audio import Audio
from HyperParams import HyperParams

hyper_params = HyperParams()
audio = Audio(hyper_params)
#input dims for model [ T_dim, num_freq ]
T_dim = 301 
num_freq =  hyper_params.num_freq
emb_dim = hyper_params.embedder_emb_dim
lstm_dim =  hyper_params.model_lstm_dim
fc1_dim = hyper_params.model_fc1_dim
fc2_dim = hyper_params.model_fc2_dim #num_freq
batch_size = 8

def get_model():
  dvec_inp = Input(shape=[emb_dim],name="dvec")
  input_spec = Input(shape=[T_dim,num_freq],name="input_spec")
  x = Reshape((T_dim,num_freq,1))(input_spec)
 
  # cnn
 
  #cnn1
  x = ZeroPadding2D(((0,0), (3,3)))(x)
  x = Conv2D(filters=64, kernel_size=[1,7], dilation_rate=[1, 1])(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  #cnn2
  x = ZeroPadding2D(((3,3), (0,0)))(x)
  x = Conv2D(filters=64, kernel_size=[7,1], dilation_rate=[1, 1])(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  #cnn3
  x = ZeroPadding2D(((2,2), (2,2)))(x)
  x = Conv2D(filters=64, kernel_size=[5,5], dilation_rate=[1, 1])(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  #cnn4
  x = ZeroPadding2D(((4,4), (2,2)))(x)
  x = Conv2D(filters=64, kernel_size=[5,5], dilation_rate=[2, 1])(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  #cnn5
  x = ZeroPadding2D(((8,8), (2,2)))(x)
  x = Conv2D(filters=64, kernel_size=[5,5], dilation_rate=[4, 1])(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  #cnn6
  x = ZeroPadding2D(((16,16), (2,2)))(x)
  x = Conv2D(filters=64, kernel_size=[5,5], dilation_rate=[8, 1])(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  #cnn7
  x = ZeroPadding2D(((32,32), (2,2)))(x)
  x = Conv2D(filters=64, kernel_size=[5,5], dilation_rate=[16, 1])(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  #cnn8
  x = Conv2D(filters=8, kernel_size=[1,1], dilation_rate=[1, 1])(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
   
  x = Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x) #else use -1 as last arg
  #x = tf.reshape(x, [x.shape[0],x.shape[1],-1])
 
  dvec = Lambda(lambda a : tf.expand_dims(a,1))(dvec_inp)
  dvec = Lambda(lambda a : tf.repeat(a,repeats =x.shape[1],axis =1))(dvec)
  #dvec= tf.expand_dims(dvec_inp,1)
  #dvec= tf.repeat(dvec,repeats =x.shape[1],axis =1)
  
  x = concatenate([x,dvec],-1)
  #x= tf.concat([x,dvec],-1)
  
  #lstm
  x = Bidirectional(LSTM(lstm_dim,return_sequences=True))(x)
  
  #fc1
  x = Dense(fc1_dim,activation ="relu")(x)
  #fc2
  mask = Dense(fc2_dim,activation ="sigmoid",name="mask")(x) #soft mask
  
  #element-wise
  output = Multiply()([input_spec,mask])

  model = Model(inputs=[input_spec,dvec_inp], outputs=output)
 
  return model