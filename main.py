from dataset import AudioDataset
from torch.utils.data import DataLoader
from model import FullModel
import warnings
import torch
from train import TrainingLoop

warnings.simplefilter(action='ignore')

PATH = './dataset/transcript.txt'
DIR = './dataset/dataset_items'

MODEL_DIR = './summary_model'

dataset = AudioDataset(data_dir=DIR,transcript_path=PATH)
tokenizer,train,val,test = dataset.get_sub_datasets()
train_loader = DataLoader(dataset=train,batch_size=16,drop_last=False)
vocab_size = train.vocab_size
val_loader = DataLoader(dataset=val,batch_size=16,drop_last=False)
test_loader = DataLoader(dataset=test,batch_size=8,drop_last=False)

model = FullModel(in_channels=2,out_channels=128,encode_dim=64,d_head=32,n_head=6,out_dim=vocab_size)
ctc_loss = torch.nn.CTCLoss(blank=0)
adam = torch.optim.Adam(params=model.parameters(),lr=1e-4)

loop = TrainingLoop(model,train_set=train_loader,val_set=val_loader,test_set=test_loader,loss=ctc_loss,dir=MODEL_DIR,optimizer=adam,tokenizer=tokenizer)
loop.run(1)



