from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import os
from utils import ctc_filter,ctc_inference
from dataset_utils import Tokenizer
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime

class TrainingLoop:
    __constants__ = ['_model','_train_set']

    def __init__(self,model:torch.nn.Module,train_set:DataLoader,val_set:DataLoader,loss=None,test_set:DataLoader=None,dir:str=None,optimizer:torch.optim.Optimizer=None,epochs=1,tokenizer:Tokenizer=None):
        self._model = model
        self._train_set = train_set
        self._val_set = val_set
        self._test_set = test_set
        self._dir = dir
        self._optimizer = optimizer
        self._epochs = epochs
        self._tokenizer = tokenizer
        self._objective_func:torch.nn.CTCLoss = loss
        if not os.path.exists(self._dir):
            os.mkdir(self._dir)
        self._timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._writer_dir = self._dir + '/summary'
        self._writer_path = self._writer_dir + f'\created_{self._timestamp}'
        self._model_dir = self._dir + '/model'
        self._writer:SummaryWriter = SummaryWriter(log_dir=self._writer_dir)

    def _compute_loss(self,logits,targets):

        batch_size = logits.shape[0]
        input_length = logits.shape[-2]
        target_length = targets.shape[-1]

        ori_log_logits = torch.log_softmax(input=logits, dim=-1)
        log_logits = torch.permute(ori_log_logits.detach().requires_grad_(), dims=(1, 0, 2))
        input_lengths = torch.full(size=(batch_size,), fill_value=input_length)
        target_lengths = torch.full(size=(batch_size,), fill_value=target_length)

        loss = self._objective_func.forward(log_probs=log_logits, targets=targets, input_lengths=input_lengths,
                                            target_lengths=target_lengths)
        if self._objective_func.reduction is None:
            loss = torch.mean(loss)
        return loss


    def run(self,epochs=None):
        epochs = self._epochs if epochs is None else epochs
        best_vloss = 100000
        for epoch in tqdm(range(epochs)):
            self._model.train(True)
            print(f'epoch: {epoch}',end='\n\t')
            average_loss = self._train_one_epoch(index_epoch=epoch)

            running_vloss = 0.0
            self._model.eval()

            with torch.no_grad():
                for i, val_batch in enumerate(self._val_set):
                    v_inputs,v_targets = val_batch
                    v_logits = self._model(v_inputs)

                    v_loss = self._compute_loss(logits=v_logits,targets=v_targets)
                    running_vloss += v_loss
            average_vloss = running_vloss / (i+1)
            print(f'LOSS train: {average_loss}, val: {average_vloss}')
            self._writer.add_scalars('Training vs. Validation Loss',
                               {'Training': average_loss, 'Validation': average_vloss},
                               epoch + 1)
            self._writer.flush()
            if average_vloss < best_vloss:
                best_vloss = average_vloss
                model_path = self._model_dir + '\model_{}_{}'.format(self._timestamp, epoch)
                torch.save(self._model.state_dict(), model_path)

    def _train_one_epoch(self,index_epoch):
        keeping_loss = 0.0
        last_loss = 0
        for i,batch in enumerate(self._train_set):
            print(f'batch: {i}',)
            inputs,targets = batch
            self._optimizer.zero_grad()

            logits = self._model(inputs)

            loss = self._compute_loss(logits=logits,targets=targets)
            self._optimizer.step()
            keeping_loss += loss

            if i % 9 == 0:
                softmax = torch.softmax(logits, dim=-1)
                print('Target: ', self._tokenizer.decode_index(ctc_filter(targets[0])[0]))
                print('Prediction: ', self._tokenizer.decode_index(ctc_inference(softmax_output=softmax, beam_width=12)[0]))

            if i % 100 == 99:
                last_loss = keeping_loss / 100
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = index_epoch * len(self._train_set) + i + 1
                self._writer.add_scalar('Loss/train', last_loss, tb_x)
                keeping_loss = 0.

        print(end='\n')
        return last_loss




