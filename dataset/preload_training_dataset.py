import os
import pandas as pd
import numpy as np

def load_col_data(data_frame ,indices, start_pos, end_pos, col):
  col_list = []
  for i in range(start_pos,end_pos):
    idx = indices[i]
    col_list.append(np.load(data_frame[col][idx]))
  return np.array(col_list)

def preload_training_data(cur_fraction, start_pos, end_pos):
  input_spec = load_col_data(df_train, list(range(num_samples)) , start_pos , end_pos ,'input_spec_path' )
  np.save(os.path.join(dataset_path,'PreLoad Training Dataset','fraction_'+str(cur_fraction),'input_spec'),input_spec)
  output_spec = load_col_data(df_train, list(range(num_samples)), start_pos, end_pos, 'output_spec_path')
  np.save(os.path.join(dataset_path,'PreLoad Training Dataset','fraction_'+str(cur_fraction),'output_spec'), output_spec)
  dvec = load_col_data(df_train, list(range(num_samples)), start_pos, end_pos, 'dvector_path')
  np.save(os.path.join(dataset_path,'PreLoad Training Dataset','fraction_'+str(cur_fraction),'dvec'),dvec)

def load_all_data(dataset):
  output_spec = load_col_data(df, list(range(df.shape[0])), 0 , df.shape[0] ,'output_spec_path')
  output_phase = load_col_data(df,list(range(df.shape[0])), 0 , df.shape[0] ,'output_phase_path')
  input_spec = load_col_data(df, list(range(df.shape[0])), 0 , df.shape[0] ,'input_spec_path')  
  input_phase = load_col_data(df,list(range(df.shape[0])), 0 , df.shape[0] ,'input_phase_path')
  dvec = load_col_data(df,list(range(df.shape[0])), 0 , df.shape[0] ,'dvector_path')
  np.save(os.path.join(path[dataset],'output_spec.npy'),output_spec)
  np.save(os.path.join(path[dataset],'output_phase.npy'),output_phase)
  np.save(os.path.join(path[dataset],'input_spec.npy'),input_spec)
  np.save(os.path.join(path[dataset],'input_phase.npy'),input_phase)
  np.save(os.path.join(path[dataset],'dvec.npy'),dvec)

dataset_train = 'train'
df_train = pd.read_csv(os.path.join(path[dataset_train],'data_frame.csv'))
num_samples = df_train.shape[0]

num_fractions = 8
fraction_sizes = num_fractions * [ num_samples//num_fractions ]
for i in range(num_samples%num_fractions):
  fraction_sizes[i]+=1
print(fraction_sizes)

start_pos = 0
for i in range(num_fractions):
    end_pos = start_pos + fraction_sizes[i]
    preload_training_data(i,start_pos, end_pos)
    start_pos = end_pos

load_all_data('dev')
load_all_data('test')
    