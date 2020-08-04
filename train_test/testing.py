import pandas as pd
import numpy as np
from mir_eval.separation import bss_eval_sources
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from model import get_model
from HyperParams import HyperParams
from Audio import Audio

hyper_params = HyperParams
audio = Audio(hyper_params)
batch_size = 8

dataset_path = os.path.join('drive','My Drive','LibriSpeech Dataset')
path = {}
path['dev'] = os.path.join(dataset_path,'LibriSpeech Dev Dataset')
path['test'] = os.path.join(dataset_path,'LibriSpeech Test Dataset')
path['train'] = os.path.join(dataset_path ,'LibriSpeech Train Dataset')
dataset = 'dev'

input_spec = np.load(os.path.join(path[dataset],'input_spec.npy'))
input_phase = np.load(os.path.join(path[dataset],'input_phase.npy'))
output_spec = np.load(os.path.join(path[dataset],'output_spec.npy'))
output_phase = np.load(os.path.join(path[dataset],'output_phase.npy'))
dvec = np.load(os.path.join(path[dataset],'dvec.npy'))

target_waves = []
for i in tqdm(range(output_spec.shape[0])):
  target_waves.append(audio.spec2wave(output_spec[i], output_phase[i]))
val_loss = []
val_sdr = []

model = get_model()
model.compile(optimizer='adam', loss='mse')

def compute_loss_sdr(weights_path):
  model.load_weights(weights_path)
  predict_spec = model.predict(x={'input_spec':input_spec,'dvec':dvec} , batch_size = batch_size, verbose = 1)
  val_loss.append(np.mean(np.square(output_spec - predict_spec)))
  sdr=[]
  for i in tqdm(range(predict_spec.shape[0])):
    #true_wave = audio.spec2wave(np.load(df_dev['output_spec_path'][i]),np.load(df_dev['output_phase_path'][i]))
    predict_wave = audio.spec2wave(predict_spec[i], input_phase[i])
    sdr.append(bss_eval_sources(target_waves[i], predict_wave, False)[0][0])
  val_sdr.append(np.median(np.array(sdr)))

### config ###
start_epochs = 1
end_epochs = 6

for i in range(start_epochs,end_epochs):
  weights_path = os.path.join(dataset_path,'Model weights','weights_epoch%04d.h5'%i)
  compute_loss_sdr(weights_path)
print()
print(val_loss)
print(val_sdr)
np.save(os.path.join(dataset_path,'val_loss_total_epochs%04d'%(end_epochs-1)),np.array(val_loss))
np.save(os.path.join(dataset_path,'val_sdr_total_epochs%04d'%(end_epochs-1)),np.array(val_sdr))

loss = np.load(os.path.join(path[dataset],'training_loss_total_epochs%04d.npy'%(end_epochs-1)))

fig , axis = plt.subplots(1,1,figsize = (9,6))
axis.plot(np.array(range(1,len(loss)+1)),loss,label="training loss")
axis.plot(5*np.array(range(len(val_loss))),val_loss,label="val loss")
axis.legend()
plt.xticks(np.array(range(len(loss)+1)))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss')
plt.show()
fig.savefig('loss.png')

fig , axis = plt.subplots(1,1,figsize = (6,6))
axis.plot(5*np.array(range(len(val_sdr))), val_sdr, label = "median sdr on val set")
plt.xlabel('epoch')
plt.ylabel('median sdr')
plt.title('SDR on dev set')
plt.show()
fig.savefig('sdr.png')