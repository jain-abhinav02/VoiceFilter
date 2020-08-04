# Voice Filter

This is a Tensorflow/Keras implementation of Google AI VoiceFilter. 

Our work is inspired from the the academic paper : https://arxiv.org/abs/1810.04826

The implementation is based on the work : https://github.com/mindslab-ai/voicefilter

---

### Team Members

1. [Angshuman Saikia](https://github.com/HeliosX7)

1. [Abhinav Jain](https://github.com/jain-abhinav02)

1. [Yashwardhan Gautam](https://github.com/yashwardhan-gautam)

---

### Introduction
We intend to improve the accuracy of Automatic speech recognition(ASR) by separating the speech of the primary speaker. 
This project has immense application in chatbots, voice assistants, video conferencing.

---

### Who is our primary speaker ?
All users of a service will have to record their voice print during enrolment. The voice print associated with the account is used to identify the primary speaker. 

### How is voice print recorded ?
A audio clip is processed by a separately trained deep neural network to generate a speaker discriminative embedding. As a result, all speakers are represented by a vector of length 256.

---

### How to prepare Dataset ?
We use the publicly available speech dataset - Librispeech. We select a primary and secondary speaker at random. For the primary speaker, select a random speech for reference and a random speech for input. Select a random speech of the secondary speaker. The input speeches of primary and secondary users are mixed which serves as one of the input. The reference speech is passed through a pre trained model ( Source: https://github.com/mindslab-ai/voicefilter ) to create an embedding which serves as the other input. The output is the input speech of the primary speaker. 
The speeches are not used directly. Instead, they are converted into magnitude spectrogram before being fed into a deep neural network. We have used python's librosa library to perform all audio related functions.

We created a dataset of 29351 samples that have been divided into 8 parts for ease of use with limited RAM. 
Link to the kaggle dataset: https://www.kaggle.com/abhinavjain02/speech-separation

---

### Stats on Prepared Data

It took around 11 hours to prepare the dataset on Google Colab. The code is present in the [dataset](/dataset) folder.

> Note: All ordered pairs of primary and secondary speakers are unique 

| Stat/Dataset                 | Train         | Dev           | Test       |
|:----------------------------:|:-------------:|:-------------:|:----------:| 
| Total no. of unique speeches available in LibriSpeech Dataset  | 28539 | 2703 | 2620 | 
| No. of unique speeches used                             | 26869 | 1878 | 1838 | 
| Percentage of total speeches used                       | 94.15 % | 69.48 % | 70.15 % | 
| Total no. of samples prepared                           | 29351 | 934 | 964 | 
| No. of samples with same primary and reference speech   | 376 (1.28 %) | 10 (1.07 %) | 11 (1.14 %) |

---

### Proposed System Architecture

<img src="/assets/images/system_arch.PNG" width="800" height="350">

---

### Requirements

*  This code was tested on Python 3.6.9 with [Google Colab](colab.research.google.com).

    Other packages can be installed by:

    ```
    pip install -r requirements.txt
    ```


### Model

<img src="/assets/images/model_workflow.PNG" width="800" height="350">

---

The model architecture is precisely as per the academic paper mentioned above. The model takes a input spectrogram and d vector(embedding) as input and produces a soft mask which when superimposed on the input spectrogram produces the output spectrogram. The output spectrogram is combined with the input phase to re create the primary speakers audio from the mixed input speech. 


| Loss Function        | Optimizer          | Metrics       |
|:--------------------:|:------------------:|:-------------:| 
| Mean Squared Error (MSE) | adam | Sound to Distortion Ratio(SDR) | 

---

<img src="/assets/images/model_plot.png" width="500" height="1300">

---

### Training

* The model was trained on Google Colab for 30 epochs.
* Training took about 37 hours on NVIDIA Tesla P100 GPU.

### Results

* Loss

<img src="/assets/images/loss.png" width="600" height="400">

* Validation SDR

<img src="/assets/images/dev_sdr.png" width="600" height="400">

* Test 

> Note: The following results are based on model weights after 29th epoch( Peak SDR on validation )

| Loss          | SDR          | 
|:-------------:|:------------:| 
| 0.0104        | 5.3250       |

---

### Audio Samples

*  Listen to the sample audio from the [assets/audio_samples](/assets/audio_samples) folder.

---

### Key learnings:

* Processing Audio data using [librosa](https://github.com/librosa/librosa)
* Creating flexible architechtures using [Keras functional API](https://keras.io/guides/functional_api/)
* Using custom generator in keras
* Using custom callbacks in keras
* Multi-Processing in python

---

### App Snippet

<img src="/assets/images/app.png" width="700" height="350">
