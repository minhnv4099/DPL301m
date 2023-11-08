import math
import os
import random
import re
import numpy as np
import torchaudio
import typing
import torch
from torch.utils.data import Dataset,IterableDataset
import librosa
from librosa import feature
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Iterable, Tuple, Iterator
from torch.utils.data.dataset import T_co
from dataset_utils import Tokenizer

PATH = './dataset/transcript.txt'
DIR = './dataset/dataset_items'

__all__ = [
    'CONFIG_INPUT',
    'CONFIG_TARGET',
    'CONFIG_DATASET',
    'AudioDataset'
]

CONFIG_INPUT = dict({
    'CHANNEL' : 2,
    'SAMPLING_RATE':44100,
    'WINDOW_LENGTH':2205,
    'WINDOW_OVERLAP':20,
    'HOP_WINDOW' : 308,

    'NFFT' : 4096,
    'N_MEL' : 224,
    'N_MFCC' : 13,

    'F_MIN':20,
    'F_MAX': 4000,

    'LIMIT_SHIFT' : 0.1,
    'MAX_MS_LENGTH' : 12000,
    'MASK_PARAM' : 0.08,
    'MASK_VALUE' : 0.0,

    'N_FRE_MASK' : 0.1,
    'N_TIME_MASK' : 0.1,

    'AMIN' : 1e-5,
    'TOP_DB' : 80,

    'PATH' :'path',
    'SAMPLE_RATE' :'sample_rate',
    'DURATION': 'duration'
})

CONFIG_TARGET = dict({
    'MAX_LENGTH': 30,
    'MASK_VALUE': 0,

    'RAW' : 'raw_transcript'
})

CONFIG_DATASET = dict({
    'CONFIG_INPUT':CONFIG_INPUT,
    'CONFIG_TARGET':CONFIG_TARGET
})


class AudioInput(Dataset):
    SAMPLING_RATE = 44100
    WINDOW_LENGTH = 2205
    WINDOW_OVERLAP = 10
    NFFT = 4096
    CHANNEL = 2
    HOP_WINDOW = 308
    N_MEL = 224
    F_MIN = 0
    F_MAX = 18000
    MAX_MS_LENGTH = 4500
    MASK_PARAM = 0.08
    MASK_VALUE = 0.0
    LIMIT_SHIFT = 0.1
    N_FRE_MASK = 0.1
    N_TIME_MASK = 0.1
    AMIN = 1e-5
    TOP_DB = 80
    N_MFCC = 40
    PATH = 'relative_path'
    SAMPLE_RATE = 'sample_rate'
    DURATION = 'duration'

    def __init__(self, list_path:pd.Series|List[str]|pd.DataFrame=None,duration_list:pd.Series|List[str]=None,config:dict=None,**kwargs):
        if config:
            self.CONFIG = config
            for k,v in config.items():
                self.__setattr__(k,v)

        self.columns = [
            self.PATH,
            self.SAMPLE_RATE,
            self.DURATION,
        ]
        self._df: pd.DataFrame = pd.DataFrame(columns=self.columns)
        if list_path is not None:
            self._ds: pd.DataFrame|pd.Series = list_path
            self._ds.columns = self.columns
            self._prepare_dataset()
        self.MAX_MS_LENGTH = kwargs['max_len'] if 'max_len' in kwargs else self.MAX_MS_LENGTH
        self.MASK_VALUE = kwargs['mask_value'] if 'mask_value' in kwargs else self.MASK_VALUE

        self._mel_args:dict = {
            'n_fft':self.NFFT,
            'win_length':self.WINDOW_LENGTH,
            'n_mels':self.N_MEL,
            'hop_length':self.HOP_WINDOW,
            'f_max':self.F_MAX,
            'f_min':self.F_MIN,
        }


    def __len__(self):
        return len(self._df)

    def __getitem__(self, index:int):
        item:pd.Series = self._df.iloc[index]
        file_path = item[self.PATH]
        duration = item[self.DURATION]
        return self._prepare_example(file_path,duration)

    def df(self,num_row:int|slice=-1) -> pd.DataFrame:
        return self._df.iloc[num_row]

    def _prepare_dataset(self):
        for i in range(len(self._ds)):
            path = self._ds.iloc[i]
            self.prepare_each(file_path_wav=path)

    def prepare_each(self,file_path_wav:str,duration:Tuple[float,float]=(0.,0.)):
        self._df = self._df._append(pd.Series(data=(file_path_wav, self.SAMPLING_RATE,duration), index=self.columns, dtype=object),ignore_index=True)

    def _prepare_example(self,file_path:str,duration:Tuple[float,float]):
        audio = self._read_wav(file_wav=file_path)
        audio = self._rechannel(audio=audio,new_channels=self.CHANNEL)
        audio = self._resample(audio,new_fs=self.SAMPLING_RATE)
        audio = self._pad_trunc(audio=audio, max_ms=self.MAX_MS_LENGTH)
        if None in audio:
            return
        specgram = self._mfcc(audio=audio)
        #masked_spec = self._spec_masking(specgram)
        return torch.permute(input=specgram,dims=(0,-1,-2))

    def _read_wav(self, file_wav: str) -> typing.Tuple[torch.Tensor,int]:
        audio_tensor, rate = torchaudio.load(file_wav)
        return audio_tensor,rate

    def _get_specific_duration(self,audio:Tuple[torch.Tensor,int],duration:Tuple[float,float]):
        signal = audio[0]
        ori_rate = audio[1]
        begin_duration,end_duration = duration[0],duration[1]
        begin_index = math.ceil(ori_rate*begin_duration)
        end_index = math.ceil(ori_rate*end_duration)
        segment_signal = signal[...,begin_index:end_index]
        return segment_signal,ori_rate

    def _rechannel(self, audio: typing.Tuple[torch.Tensor, int],new_channels: int = 1) \
            -> typing.Tuple[torch.Tensor,int]:
        signal, rate = audio
        if signal.shape[0] == new_channels:
            return audio
        if new_channels == 1:
            re_channeled_signal = signal[:1, :]
        else:
            re_channeled_signal = torch.cat(tensors=(signal, signal),dim=0)
        return re_channeled_signal, rate

    def _resample(self, audio: typing.Tuple[torch.Tensor, int], new_fs: int = None) \
            -> typing.Tuple[torch.Tensor,int]:
        signal, rate = audio
        if new_fs == rate:
            return signal,rate

        channels = signal.shape[0]
        resample_transfer = torchaudio.transforms.Resample(orig_freq=rate, new_freq=new_fs)
        re_sampled_signal = resample_transfer(signal[:1, :])


        if channels > 1:
            re_sampled_signal_two = resample_transfer(signal[1:, :])
            re_sampled_signal = torch.cat(tensors=(re_sampled_signal, re_sampled_signal_two),dim=0)
        return re_sampled_signal, new_fs

    def _pad_trunc(self, audio: typing.Tuple[torch.Tensor, int], max_ms: int = None) -> any:
        signal, rate = audio
        n_channels, len_signal = signal.shape
        max_len = (max_ms // 1000) * rate

        if max_len <= len_signal:
            return None,rate
        else:
            head_pad_end = random.randint(a=0, b=max_len - len_signal)
            tail_pad_start = max_len - len_signal - head_pad_end

            head_padding = torch.zeros(size=(n_channels, head_pad_end))
            tail_padding = torch.zeros(size=(n_channels, tail_pad_start))

            paded_signal = torch.cat(tensors=(head_padding, signal, tail_padding), dim=-1)
            return paded_signal, rate

    def _time_shift(self, audio: typing.Tuple[torch.Tensor, int],limit_shift: float = None) \
            -> typing.Tuple[torch.Tensor,int]:
        signal, rate = audio
        len_signal = signal.shape[1]
        shift_amt = int(random.random() * limit_shift * len_signal)
        shifted_time_signal = torch.roll(input=signal, shifts=shift_amt)
        return shifted_time_signal,rate

    def _shift_pitch(self, audio: typing.Tuple[torch.Tensor, int]) \
            -> typing.Tuple[torch.Tensor,int]:
        signal, rate = audio

        pitch_shift_transfer = torchaudio.transforms.PitchShift(sample_rate=rate)
        pitch_shifted_signal = pitch_shift_transfer.forward(waveform=signal)
        return pitch_shifted_signal, rate

    def _convert_to_frequency(self,audio:typing.Tuple[torch.Tensor,int],n_fft=200):
        signal,rate = audio
        squeez_signal = torch.squeeze_copy(signal)
        t = torch.squeeze_copy(torch.linspace(start=0, end=len(signal)/rate, steps=len(squeez_signal)))
        dft = torch.fft.rfft(squeez_signal,n=n_fft)
        dft_amplitude = torch.abs(dft)
        freqs = torch.fft.rfftfreq(n=n_fft, d=1. / rate)
        f,ax = plt.subplots(2,1,figsize=(12,7))
        ax[0].plot(t,torch.squeeze_copy(signal))
        ax[1].plot(freqs,dft_amplitude)
        plt.show()

    def _specgram(self, audio: typing.Tuple[torch.Tensor, int],plot_:bool=False) -> any:
        signal, rate = audio
        if signal is None:
            return
        specgram_transfer = torchaudio.transforms.Spectrogram(n_fft=self.NFFT, win_length=self.WINDOW_LENGTH,hop_length=self.HOP_WINDOW)
        specgram_complex = specgram_transfer(signal)
        specgram_amplitude = torch.abs(specgram_complex)
        specgram_db = librosa.power_to_db(specgram_amplitude.numpy(),ref=1.0,top_db=self.TOP_DB,amin=self.AMIN)
        return torch.from_numpy(specgram_db)


    def _mel_specgram(self, audio: typing.Tuple[torch.Tensor, int]) -> any:
        signal, rate = audio
        if signal is None:
            return
        mel_specgram_transfer = torchaudio.transforms.MelSpectrogram(sample_rate=self.SAMPLING_RATE,**self._mel_args)
        mel_specgram_amplitude = mel_specgram_transfer.forward(signal)
        mel_specgram_db = librosa.power_to_db(mel_specgram_amplitude.numpy(),ref=np.max,amin=self.AMIN,top_db=self.TOP_DB)
        return torch.from_numpy(mel_specgram_db)

    def _mfcc(self, audio: typing.Tuple[torch.Tensor, int]) -> any:
        signal, rate = audio
        if signal is None:
            return
        mfcc_transfer = torchaudio.transforms.MFCC(n_mfcc=self.N_MFCC,sample_rate=self.SAMPLING_RATE,melkwargs=self._mel_args)
        mfcc_amplitude = mfcc_transfer.forward(waveform=signal)
        mfcc_db = librosa.power_to_db(mfcc_amplitude.numpy(),ref=np.max,amin=self.AMIN,top_db=self.TOP_DB)
        return torch.from_numpy(mfcc_db)

    def _spec_masking(self, spec: torch.Tensor, max_mask_p=0.2, n_freqs_mask: int = 2, n_time_mask: int = 2) -> any:
        if spec is None:
            return
        _, n_fft, n_time = spec.shape
        mask_value = torch.mean(spec).item()
        aug_spec = spec

        fre_mask_param = int(max_mask_p * n_fft)
        for _ in range(n_freqs_mask):
            aug_spec = torchaudio.functional.mask_along_axis(specgram=spec, mask_param=fre_mask_param,mask_value=mask_value, axis=1)

        time_mask_param = int(max_mask_p * n_time)
        for _ in range(n_time_mask):
            aug_spec = torchaudio.functional.mask_along_axis(specgram=spec, mask_param=time_mask_param,mask_value=mask_value, axis=2)
        return aug_spec


    def _plot_specgram(self,items:typing.Tuple[torch.Tensor,torch.Tensor,torch.Tensor,int]):
        len_items = len(items)
        figure,ax = plt.subplots(nrows=len_items,ncols=1,figsize=(12,7))
        for i,item in enumerate(items):
            specgram = item[0]
            f = torch.arange(specgram.shape[-2])
            t = torch.arange(specgram.shape[-1])
            if len(specgram) == 2: specgram = specgram[0]
            ax[i].pcolormesh(t,f,specgram,cmap='inferno')
            #librosa.display.specshow(specgram.numpy(),x_axis='time',y_axis='hz',ax=ax)
        plt.show()

class TransciptTarget(Dataset):
    MAX_LENGTH = 200
    MASK_VALUE = 0
    RAW = 'raw'

    def __init__(self,sequence_of_transcript:pd.Series|List[str]|Iterable[str]=None,config:dict=None,tokenizer:Tokenizer=None,**kwargs):
        if config:
            self.CONFIG = config
            for k,v in self.CONFIG.items():
                self.__setattr__(k,v)

        self.columns = [
            self.RAW
        ]

        self._tokenizer = Tokenizer(max_len=self.MAX_LENGTH)
        self._is_learn = False
        if tokenizer is not None:
            self._tokenizer = tokenizer
            self._is_learn = True

        self._df: pd.DataFrame = pd.DataFrame(columns=self.columns)
        if sequence_of_transcript is not None:
            self._ds:pd.Series|pd.DataFrame = sequence_of_transcript
            self._ds.columns = self.columns
            self._prepare_dataset()

    @property
    def tokenizer(self):
        return self._tokenizer

    def prepare_each(self,raw_transcript:str):
        if self._is_learn is False:
            self._tokenizer.update_base_sentence(sentence=raw_transcript)
        self._df = self._df._append(pd.Series(data=(raw_transcript),index=self.columns),ignore_index=True)

    def _prepare_dataset(self):
        for i in range(len(self._ds)):
            self.prepare_each(self._ds.iloc[i])
        self.learn()
        del self._ds

    def learn(self):
        if not self._is_learn:
            self._tokenizer.learn_bpe()
            self._is_learn = True

    def _prepare_example(self,raw_transcript:str):
        token = self._tokenizer.encode_as_token(sentence=raw_transcript)
        if token is None:
            return None
        target = torch.tensor(self._tokenizer.encode_as_index(sentence=raw_transcript),dtype=torch.int8)
        mask = torch.where(target == self._tokenizer.get_index(self._tokenizer.PAD), target, self.MASK_VALUE)
        return target

    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index:int):
        raw_transcript = self._df.iloc[index][self.RAW]
        return self._prepare_example(raw_transcript=raw_transcript)

    def df(self,num_row:int|slice=-1):
        return self._df.iloc[num_row]

class AudioDataset(Dataset):

    FILE_WAV = 'file_name'
    PATH_FILE_WAV = 'file_path'
    TRANSCRIPTS = 'transcript'
    DURATION = 'duration(s)'

    def __init__(self,data_dir:str=DIR,transcript_path:str=PATH,config:dict=CONFIG_DATASET,num_examples=-1,splits:Tuple=(0.7,0.15,0.15),**kwargs):

        self.CONFIG_DATASET:dict = config

        self.CONFIG_INPUT: dict
        self.CONFIG_TARGET: dict

        if 'CONFIG_INPUT' not in config:
            raise KeyError(f'Missing keyword "CONFIG_INPUT"')
        if 'CONFIG_TARGET' not in config:
            raise KeyError(f'Missing keyword "CONFIG_TARGET"')

        for k,v in self.CONFIG_DATASET.items():
            self.__setattr__(k,v)

        self.data_dir = data_dir
        self.transcript_path = transcript_path
        self.ds_columns = [
            self.FILE_WAV,
            self.TRANSCRIPTS,
            self.DURATION,
        ]
        self._ds: pd.DataFrame = pd.read_csv(self.transcript_path, delimiter='|').iloc[:num_examples]
        self._ds.columns = self.ds_columns
        self._ds = self._ds.sample(frac=1)

        self._splits = ['train', 'val', 'test']
        self._train_length, self._valid_length, self._test_length = [int(len(self) * frac) for frac in splits]

        self._train_ds: pd.DataFrame = self._ds.iloc[:self._train_length]
        self._val_ds: pd.DataFrame = self._ds.iloc[self._train_length:self._train_length+self._valid_length]
        self._test_ds: pd.DataFrame = self._ds.iloc[-self._valid_length:-1]

        self._inputs: AudioInput = AudioInput(config=self.CONFIG_INPUT)
        self._targets: TransciptTarget = TransciptTarget(config=self.CONFIG_TARGET)
        self._prepare_dataset()

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, index:int):
        pair_example = (self._inputs[index],self._targets[index])
        return pair_example

    def get_sub_datasets(self,sub_set:Tuple=('train','val','test')):

        self._training_dataset = SubAudioDataset(d=self.get_d(self._train_ds),config=self.CONFIG_DATASET)
        self._validation_dataset = SubAudioDataset(d=self.get_d(self._val_ds),config=self.CONFIG_DATASET,tokenizer=self._training_dataset.tokenizer)
        self._testing_dataset = SubAudioDataset(d=self.get_d(self._test_ds),config=self.CONFIG_DATASET,tokenizer=self._training_dataset.tokenizer)

        self.sub_datasets = {
            'train': AudioIterableDataset(dataset=self._training_dataset),
            'val': AudioIterableDataset(dataset=self._validation_dataset),
            'test': AudioIterableDataset(dataset=self._testing_dataset),
        }

        return self._training_dataset.tokenizer,*(self.sub_datasets[sub] for sub in sub_set)

    def df(self,num_row:slice):
        d = dict()
        d['inputs'] = self._inputs.df(num_row=num_row)
        d['targets'] = self._targets.df(num_row=num_row)
        return d

    def get_d(self,df:pd.DataFrame):
        d = dict()
        d['inputs'] = pd.Series((os.path.join(self.data_dir,file_wav) for file_wav in df[self.FILE_WAV]))
        d['targets'] = df[self.TRANSCRIPTS]
        return d

    @property
    def vocab_size(self):
        return self._targets.vocab_size

    def _prepare_dataset(self):
        for row in self._ds.iterrows():
            file_path:str = os.path.join(self.data_dir,row[1][self.FILE_WAV])
            duration:str = row[1][self.DURATION]
            split_duration = re.split(pattern=r'[\-+\s+\D+]',string=duration)
            end_duration = float(split_duration[-1])
            durations = (0,end_duration)
            transcipt = row[1][self.TRANSCRIPTS]
            self._inputs.prepare_each(file_path_wav=file_path,duration=durations)
            self._targets.prepare_each(raw_transcript=transcipt)
        self._targets.learn()

class SubAudioDataset(Dataset):
    def __init__(self,d:dict,config:dict,tokenizer:Tokenizer=None):
        self._d = d

        self._inputs = AudioInput(self._d['inputs'],config=config['CONFIG_INPUT'])
        self._targets = TransciptTarget(self._d['targets'],tokenizer=tokenizer,config=config['CONFIG_TARGET'])

    def __len__(self):
        return len(self._inputs)

    def __getitem__(self,index:int):
        pair = (self._inputs[index],self._targets[index])
        return pair

    @property
    def inputs(self):
        return self._inputs

    @property
    def tokenizer(self):
        return self._targets.tokenizer

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

class AudioIterableDataset(IterableDataset):

    def __init__(self,dataset:Dataset):
        self._dataset = dataset
        self._iter_dataset = iter(dataset)

    def __iter__(self) -> Iterator[T_co]:
        return self

    def __next__(self):
        item = next(self._iter_dataset)
        if None in item:
            return next(self)
        return item

    @property
    def vocab_size(self):
        return self._dataset.vocab_size

    def __len__(self):
        return len(self._dataset)

if __name__ == '__main__':
    pass

