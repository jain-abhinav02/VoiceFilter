from tensorflow.keras.utils import Sequence
import numpy as np
import os

dataset_path = os.path.join('drive','My Drive','LibriSpeech Dataset')
path = {}
path['dev'] = os.path.join(dataset_path,'LibriSpeech Dev Dataset')
path['test'] = os.path.join(dataset_path,'LibriSpeech Test Dataset')
path['train'] = os.path.join(dataset_path ,'LibriSpeech Train Dataset')


class data_generator(Sequence):
  def __init__(self, fraction_sizes , batch_size):
    self.batch_size = batch_size
    self.fraction_sizes = fraction_sizes
    self.num_fractions = len(fraction_sizes)
    self.num_samples = sum(self.fraction_sizes)
    self.pos = 0
    self.cur_fraction = 0
    self.input_spec=None
    self.output_spec=None
    self.dvec=None
  def __len__(self):
    self.num_batches = 0
    for i in range(self.num_fractions):
      self.num_batches += (self.fraction_sizes[i]+self.batch_size-1)//self.batch_size
    print("len ",self.num_batches)
    return self.num_batches
  def __getitem__(self,batch_index):
    start_pos = self.pos
    end_pos = start_pos + self.batch_size
    if end_pos > self.fraction_sizes[self.cur_fraction]:
      end_pos = self.fraction_sizes[self.cur_fraction]
    if start_pos == 0 :
      print("loading new data")
      if self.input_spec is not None:
        print("de allocating old space")
        del self.input_spec
        del self.output_spec
        del self.dvec
      indices = np.random.permutation(self.fraction_sizes[self.cur_fraction])
      self.input_spec = np.load(os.path.join(dataset_path,'PreLoad Training Dataset','fraction_'+str(self.cur_fraction),'input_spec.npy'))[indices]
      self.output_spec = np.load(os.path.join(dataset_path,'PreLoad Training Dataset','fraction_'+str(self.cur_fraction),'output_spec.npy'))[indices]
      self.dvec = np.load(os.path.join(dataset_path,'PreLoad Training Dataset','fraction_'+str(self.cur_fraction),'dvec.npy'))[indices]
      print("new data loaded ")
    if end_pos == self.fraction_sizes[self.cur_fraction]:
      self.cur_fraction += 1
      if self.cur_fraction == self.num_fractions:
        self.cur_fraction = 0
      self.pos = 0
    else :
      self.pos = end_pos
    input_spec_batch = self.input_spec[start_pos:end_pos]
    dvector_batch = self.dvec[start_pos:end_pos]
    output_spec_batch = self.output_spec[start_pos:end_pos]
    return ({'input_spec':input_spec_batch, 'dvec': dvector_batch}, output_spec_batch)
  def on_epoch_end(self):
    self.pos = 0
    self.cur_fraction = 0