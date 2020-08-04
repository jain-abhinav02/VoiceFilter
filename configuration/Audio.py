import librosa
import numpy as np

class Audio:
  def __init__(self,hyper_params):
    self.hyper_params = hyper_params
    self.mel_basis_matrix = librosa.filters.mel(sr=hyper_params.sample_rate,
                                             n_fft=hyper_params.n_fft,
                                             n_mels=hyper_params.embedder_num_mels)

  def get_mel_spec(self,wave):
    spec = librosa.core.stft(y=wave, n_fft=self.hyper_params.n_fft,
                              hop_length=self.hyper_params.hop_length,
                              win_length=self.hyper_params.win_length,
                              window='hann')
    power_spec = np.abs(spec) ** 2
    mel_spec = np.log10(np.dot(self.mel_basis_matrix,power_spec)+1e-6)
    return mel_spec  
  def wave2spec(self,wave): 
    spec = librosa.core.stft(y=wave, n_fft=self.hyper_params.n_fft,
                            hop_length=self.hyper_params.hop_length,
                            win_length=self.hyper_params.win_length)
    phase = np.angle(spec)
    spec_db = self.amp2db(np.abs(spec))
    spec_db_norm = self.normalize(spec_db)
    spec_db_norm = spec_db_norm.T   # Taking transpose here
    phase = phase.T # Taking transpose here
    return spec_db_norm, phase
  def spec2wave(self,spec_db_norm,phase):
    spec_db_norm, phase = spec_db_norm.T, phase.T
    spec_db = self.denormalize(spec_db_norm)
    spec_amp = self.db2amp(spec_db)
    spec = spec_amp * np.exp(1j*phase)
    wave = librosa.core.istft(spec,
                             hop_length=self.hyper_params.hop_length,
                             win_length=self.hyper_params.win_length)
    return wave
  def amp2db(self,mat):
    return 20.0 * np.log10(np.maximum(1e-5,mat)) - self.hyper_params.ref_level_db
  def db2amp(self,mat):
    return np.power(10.0, (mat+self.hyper_params.ref_level_db)*0.05)
  def normalize(self,mat):
    return np.clip((mat-self.hyper_params.min_level_db)/-self.hyper_params.min_level_db, 0.0 , 1.0)
  def denormalize(self, mat):
    return np.clip(mat,0.0,1.0)*(-self.hyper_params.min_level_db)+self.hyper_params.min_level_db