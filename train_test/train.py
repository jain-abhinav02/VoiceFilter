import os
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import tqdm

from Audio import Audio
from HyperParams import HyperParams
from model import get_model
from sequence_generator import data_generator

hyper_params = HyperParams()
audio = Audio(hyper_params)
dataset_path = os.path.join('drive','My Drive','LibriSpeech Dataset')
path = {}
path['dev'] = os.path.join(dataset_path,'LibriSpeech Dev Dataset')
path['test'] = os.path.join(dataset_path,'LibriSpeech Test Dataset')
path['train'] = os.path.join(dataset_path ,'LibriSpeech Train Dataset')

model = get_model()
model.summary()

model.compile(optimizer='adam', loss='mse')

### execute just once ###
model.save_weights(os.path.join(dataset_path,'Model weights','weights_epoch0000.h5'))

dataset_train = 'train'
df_train = pd.read_csv(os.path.join(path[dataset_train],'data_frame.csv'))
num_samples = df_train.shape[0]
batch_size = 8

num_fractions = 8
fraction_sizes = num_fractions * [ num_samples//num_fractions ]
for i in range(num_samples%num_fractions):
  fraction_sizes[i]+=1
print(fraction_sizes)

steps_per_epoch = 0
for i in range(num_fractions):
  steps_per_epoch += (fraction_sizes[i]+batch_size-1)//batch_size
print(steps_per_epoch)

gen_train = data_generator(fraction_sizes, batch_size)

model_checkpoint_callback = ModelCheckpoint(
    os.path.join(dataset_path,'Model weights','weights_epoch{epoch:04d}.h5'),save_weights_only=True)

### config ###
initial_epoch = 0
epochs = 20

hist = model.fit(gen_train, epochs=epochs, verbose=1, steps_per_epoch = 2, shuffle=False,
          callbacks=[model_checkpoint_callback 
#                     myCallback() 
                     ]
          )

loss = hist.history['loss']
np.save(os.path.join(dataset_path,'training_loss_total_epochs%04d'%epochs),np.array(loss))
