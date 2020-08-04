import os
import pickle
import torch
import torch.nn as nn
import random
import pandas as pd
import time
import numpy as np
import librosa
from multiprocessing import Pool
from Audio import Audio
from HyperParams import HyperParams

class LinearNorm(nn.Module):
    def __init__(self, hp):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(hp.embedder_lstm_hidden, hp.embedder_emb_dim)

    def forward(self, x):
        return self.linear_layer(x)

class SpeechEmbedder(nn.Module):
    def __init__(self, hp):
        super(SpeechEmbedder, self).__init__()
        self.lstm = nn.LSTM(hp.embedder_num_mels,
                            hp.embedder_lstm_hidden,
                            num_layers=hp.embedder_lstm_layers,
                            batch_first=True)
        self.proj = LinearNorm(hp)
        self.hp = hp

    def forward(self, mel):
        # (num_mels, T)
        mels = mel.unfold(1, self.hp.embedder_window, self.hp.embedder_stride) # (num_mels, T', window)
        mels = mels.permute(1, 2, 0) # (T', window, num_mels)
        x, _ = self.lstm(mels) # (T', window, lstm_hidden)
        x = x[:, -1, :] # (T', lstm_hidden), use last frame only
        x = self.proj(x) # (T', emb_dim)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True) # (T', emb_dim)
        x = x.sum(0) / x.size(0) # (emb_dim), average pooling over time frames
        return x

def get_dvector(wave):
  mel_spec = audio.get_mel_spec(wave)
  dvec = embedder(torch.from_numpy(mel_spec).float())
  dvec = dvec.detach().numpy()
  return dvec

def create_example(target_dir, hyper_params, idx, ref_speech, pri_speech, sec_speech):
  sample_rate = hyper_params.sample_rate
  ref_wave, _ = librosa.load(ref_speech,sr=sample_rate) #load the audio file
  pri_wave, _ = librosa.load(pri_speech, sr = sample_rate)
  sec_wave,_ = librosa.load(sec_speech, sr = sample_rate)
  assert len(ref_wave.shape)==len(pri_wave.shape)==len(sec_wave.shape)==1,\
  'wave files must be mono and not stereo'
  ref_wave,_ = librosa.effects.trim(ref_wave, top_db = 20) # clip silent portion on both ends
  pri_wave,_ = librosa.effects.trim(pri_wave, top_db = 20)
  sec_wave,_ = librosa.effects.trim(sec_wave, top_db = 20)
  
  if ref_wave.shape[0] < 1.1 * hyper_params.embedder_window * hyper_params.hop_length :
    return
  length_wave = int(sample_rate * hyper_params.data_audio_len)
  if pri_wave.shape[0]<length_wave or sec_wave.shape[0]<length_wave:
    return
  pri_wave, sec_wave = pri_wave[:length_wave], sec_wave[:length_wave] # clip wave to fixed length
  mix_wave = pri_wave + sec_wave
  norm = np.max(np.abs(mix_wave)) * 1.1
  pri_wave, mix_wave = pri_wave/norm , mix_wave/norm  # normalize wave by 1.1*max(absolute amplitude)
  pri_spec, pri_phase = audio.wave2spec(pri_wave)  # convert wave to spec
  mix_spec, mix_phase = audio.wave2spec(mix_wave)
  dvec = get_dvector(ref_wave)

  # paths for storing data
  pri_spec_path = os.path.join(target_dir,'output_spec','%06d.npy'%idx)
  pri_phase_path = os.path.join(target_dir,'output_phase','%06d.npy'%idx)
  mix_spec_path = os.path.join(target_dir, 'input_spec','%06d.npy'%idx)
  mix_phase_path = os.path.join(target_dir,'input_phase','%06d.npy'%idx)
  dvec_path = os.path.join(target_dir,'dvec','%06d.npy'%idx)
  # store data on paths above
  np.save(pri_spec_path,pri_spec)
  np.save(mix_spec_path,mix_spec)
  np.save(mix_phase_path, mix_phase)
  np.save(pri_phase_path, pri_phase)
  np.save(dvec_path,dvec)

  #print(idx)
  return [idx, ref_speech, pri_speech, sec_speech,  mix_spec_path, pri_spec_path, mix_phase_path, pri_phase_path, dvec_path]

def create_dataset(i):
  batch = []
  array = range(i+1,n+1)
  if parity == 1:
    array = range(1,i)
  for j in array:
    first = min(i,j)
    sec = max(i,j)
    if (sec-first)%2 == parity:
      first, sec = sec, first
    n1 = len(speeches[first-1]) # -1 accounts for zero based indexing
    n2 = len(speeches[sec-1]) # -1 accounts for zero based indexing
    sum = first+sec-1 # -1 accounts for zero based indexing
    diff = first-sec-1 # -1 accounts for zero based indexing
    diff_mod = (abs(diff))%n1
    if diff < 0 and diff_mod > 0:
      diff_mod = n1 - diff_mod
    ref_speech = speeches[first-1][diff_mod]
    pri_speech = speeches[first-1][sum%n1]
    sec_speech = speeches[sec-1][first%n2]
    row = create_example( path[dataset], hyper_params , n*(i-1) + j, ref_speech, pri_speech, sec_speech)
    if row is not None:
      batch.append(row)
  print(i)
  return batch

def save_batch(dataset,data):
  df_path = os.path.join(path[dataset],'data_frame.csv')
  df = pd.read_csv(df_path)
  df_batch = pd.DataFrame(data = data, columns = columns)
  df = df.append(df_batch)
  df.to_csv(df_path,index=False)

columns=['key','ref_speech','pri_speech','sec_speech','input_spec_path','output_spec_path','input_phase_path','output_phase_path','dvector_path']
hyper_params = HyperParams()
audio = Audio(hyper_params)
dataset_path = os.path.join('drive','My Drive','LibriSpeech Dataset')
path = {}
path['dev'] = os.path.join(dataset_path,'LibriSpeech Dev Dataset')
path['test'] = os.path.join(dataset_path,'LibriSpeech Test Dataset')
path['train'] = os.path.join(dataset_path ,'LibriSpeech Train Dataset')

with open(os.path.join(path['dev'],'dev_speeches.data'),'rb') as f:
  dev_speeches = pickle.load(f)
with open(os.path.join(path['test'],'test_speeches.data'),'rb') as f:
  test_speeches = pickle.load(f)
with open(os.path.join(path['train'],'train_speeches.data'),'rb') as f:
  train_speeches = pickle.load(f)

embedder_path = os.path.join(dataset_path,"embedder.pt")
embedder_pt = torch.load(embedder_path,map_location=torch.device('cpu'))
embedder = SpeechEmbedder(hyper_params)
embedder.load_state_dict(embedder_pt)
embedder.eval()

cpu_num = len(os.sched_getaffinity(0))

#### Dev Dataset ####
dataset = 'dev' # important global variable
speeches = dev_speeches # important global variable
n = len(dev_speeches) # important global variable
print("number of speakers(dev set) : ",n)
for i in range(n):
  random.shuffle(dev_speeches[i])  # shuffle the speeches of all speakers
arr = list(range(1,n+1))  # create a list for all speakers

data = []
parity = 0 # important global variable
with Pool(cpu_num) as p:
  res = p.map(create_dataset, arr , 4)
for batch in res:
  if len(batch) > 0:
    data.extend(batch)
save_batch('dev',data)

data = []
parity = 1 # important global variable
with Pool(cpu_num) as p:
  res = p.map(create_dataset, arr , 4)
for batch in res:
  if len(batch) > 0:
    data.extend(batch)
save_batch('dev',data)

#### Test dataset ####
dataset = 'test' # important global variable
speeches = test_speeches # important global variable
n = len(test_speeches) # important global variable
print("number of speakers(test set) : ",n)
for i in range(n):
  random.shuffle(test_speeches[i])  # shuffle the speeches of all speakers
arr = list(range(1,n+1))  # create a list for all speakers

data = []
parity = 0 # important global variable
x = time.time()
with Pool(cpu_num) as p:
  res = p.map(create_dataset, arr , 4)
for batch in res:
  if len(batch) > 0:
    data.extend(batch)
y = time.time()
print(y-x)
save_batch('test',data)

data = []
parity = 1 # important global variable
x = time.time()
with Pool(cpu_num) as p:
  res = p.map(create_dataset, arr , 4)
for batch in res:
  if len(batch) > 0:
    data.extend(batch)
y = time.time()
print(y-x)
save_batch('test',data)

#### Train dataset ####
dataset = 'train' # important global variable
speeches = train_speeches # important global variable
n = len(train_speeches) # important global variable
print("number of speakers(train set) : ",n)
for i in range(n):
  random.shuffle(train_speeches[i])  # shuffle the speeches of all speakers
arr = list(range(1,n+1))  # create a list for all speakers

data = []
parity = 0 # important global variable
with Pool(cpu_num) as p:
  res = p.map(create_dataset, arr , 4)
for batch in res:
  if len(batch) > 0:
    data.extend(batch)
save_batch('train',data)

### Run the following code only to create additional 30k samples for training set
#data = []
#parity = 1 # important global variable
#with Pool(cpu_num) as p:
#  res = p.map(create_dataset, arr , 4)
#for batch in res:
#  if len(batch) > 0:
#    data.extend(batch)
#save_batch('train',data)


