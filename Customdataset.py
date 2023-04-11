from torch.utils.data import Dataset
from torch.nn import functional as F
import random
import torch
import numpy as np
import pandas as pd
import soundfile as sf
import whisper
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Klue/RoBERTa-large")

N_SAMPLES = 480000

"""기쁨happy, 놀람surprise, 분노angry, 중립neutral, 혐오disgust, 공포fear, 슬픔sad"""

def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array

def collate_fn(data):
    outputs = {}
    for key, value in zip(("text_tokens","labels","wav_tokens"),zip(*data)):
        outputs[key]=value
    outputs["text_tokens"] = tokenizer(outputs["text_tokens"],return_tensors="pt",padding=True)
    outputs["wav_tokens"] = torch.concat([i.unsqueeze(0) for i in outputs["wav_tokens"]])
    return outputs
    

class CustomDataset(Dataset):
    def __init__(self, csv_path, processor=None):
        self.data = pd.read_csv(csv_path, header=[0, 1])
        self.text_data = self.data['text_data'][' '].values
        self.wav_dir = self.data['wav_dir'][' '].values
        self.dic = {'happy': 0, 'surprise': 1, 'angry': 2, 'neutral': 3, 'disqust': 4, 'fear': 5, 'sad': 6}
        self.labels = self.data['Total Evaluation']['Emotion'].values
        self.processor = processor
        self.shuffle = np.arange(len(self))
        np.random.shuffle(self.shuffle)
      
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        idx = self.shuffle[idx]
        if ';' in self.labels[idx]:
            text_label = self.labels[idx].split(';')[random.choice([0,1])]
        else:
            text_label = self.labels[idx]

        if self.processor:
            # in whisper
            # audio_input, sample_rate = sf.read(self.wav_dir[idx])
            # audio_input = pad_or_trim(audio_input)
            # audio_input_values = self.processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values.squeeze(0)
            audio_input = whisper.load_audio(self.wav_dir[idx])
            audio_input = whisper.pad_or_trim(audio_input)  
            audio_input_values = whisper.log_mel_spectrogram(audio_input)
            return self.text_data[idx], self.dic[text_label], audio_input_values
        
        else:
            return self.text_data[idx], self.dic[text_label]


