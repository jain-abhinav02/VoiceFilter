import os
import pandas as pd
import shutil

dataset_path = os.path.join('drive','My Drive','LibriSpeech Dataset')
path = {}
path['dev'] = os.path.join(dataset_path,'LibriSpeech Dev Dataset')
path['test'] = os.path.join(dataset_path,'LibriSpeech Test Dataset')
path['train'] = os.path.join(dataset_path ,'LibriSpeech Train Dataset')

shutil.unpack_archive(os.path.join(dataset_path,'dev-clean.tar.gz'),os.path.join(dataset_path))
# Rename the extracted folder LibriSpeech to LibriSpeech Dev Dataset
shutil.unpack_archive(os.path.join(dataset_path,'test-clean.tar.gz'),os.path.join(dataset_path))
# Rename the extracted folder LibriSpeech to LibriSpeech Test Dataset
shutil.unpack_archive(os.path.join(dataset_path,'train-clean-100.tar.gz'),os.path.join(dataset_path))
# Rename the extracted folder LibriSpeech to LibriSpeech Train Dataset

for dataset in ('dev','test'):
  os.makedirs(os.path.join(path[dataset],'input_spec'),exist_ok=True)
  os.makedirs(os.path.join(path[dataset],'output_spec'),exist_ok=True)
  os.makedirs(os.path.join(path[dataset],'input_phase'),exist_ok=True)
  os.makedirs(os.path.join(path[dataset],'output_phase'),exist_ok=True)
  os.makedirs(os.path.join(path[dataset],'dvector'),exist_ok=True)

def create_folders(i):
  os.makedirs(os.path.join(path['train'],'input_spec_'+i),exist_ok=True)
  os.makedirs(os.path.join(path['train'],'output_spec_'+i),exist_ok=True)
  os.makedirs(os.path.join(path['train'],'input_phase_'+i),exist_ok=True)
  os.makedirs(os.path.join(path['train'],'output_phase_'+i),exist_ok=True)
  os.makedirs(os.path.join(path['train'],'dvector_'+i),exist_ok=True)

for i in range(8):
  create_folders(str(i))

columns=['key','ref_speech','pri_speech','sec_speech','input_spec_path','output_spec_path','input_phase_path','output_phase_path','dvector_path']
sample_data_frame = pd.DataFrame(data = [], columns=columns)
for dataset in ('train','dev','test'):
  sample_data_frame.to_csv(os.path.join(path[dataset],'data_frame.csv'),index=False)