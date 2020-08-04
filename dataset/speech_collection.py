import glob
import pickle
import os
dataset_path = os.path.join('drive','My Drive','LibriSpeech Dataset')
path = {}
path['dev'] = os.path.join(dataset_path,'LibriSpeech Dev Dataset')
path['test'] = os.path.join(dataset_path,'LibriSpeech Test Dataset')
path['train'] = os.path.join(dataset_path ,'LibriSpeech Train Dataset')

#### Run this cell only the first time ####
dev_base_path = os.path.join(path['dev'],'dev-clean')
test_base_path = os.path.join(path['test'],'test-clean')
train_base_path = os.path.join(path['train'],'train-clean-100')

#### Run this cell only the first time ####
dev_spks = os.listdir(dev_base_path)
# list of all speaker folders
dev_speeches = [glob.glob(os.path.join(dev_base_path,spk,'*','*.flac'),recursive=True) for spk in dev_spks]
dev_speeches = [speeches for speeches in dev_speeches if len(speeches)>=2]
# list of lists containing speeches of a speaker
test_spks = os.listdir(test_base_path)
# list of all speaker folders
test_speeches = [glob.glob(os.path.join(test_base_path,spk,'*','*.flac'),recursive=True) for spk in test_spks]
test_speeches = [speeches for speeches in test_speeches if len(speeches)>=2]
# list of lists containing speeches of a speaker
train_spks = os.listdir(train_base_path)
# list of all speaker folders
train_speeches = [glob.glob(os.path.join(train_base_path,spk,'*','*.flac'),recursive=True) for spk in train_spks]
train_speeches = [speeches for speeches in train_speeches if len(speeches)>=2]
# list of lists containing speeches of a speaker

##### Run this cell only the first time #####
with open(os.path.join(path['dev'],'dev_speeches.data'),'wb') as f:
  pickle.dump(dev_speeches,f)
with open(os.path.join(path['test'],'test_speeches.data'),'wb') as f:
  pickle.dump(test_speeches,f)
with open(os.path.join(path['train'],'train_speeches.data'),'wb') as f:
  pickle.dump(train_speeches,f)

