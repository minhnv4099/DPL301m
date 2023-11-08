from typing import Tuple,List
import re
import string
from tqdm import tqdm
from collections import defaultdict,Counter,OrderedDict

CONFIG_TOKENIZER = dict({
    'UNK' : '<UNKONW>',
    'PAD' :'<PAD>',
    'FRACTION' : 0,  # -> character level
    'OPTIMAL_LENGTH' : 10,
    'OPTION_PAD' : 'post'
})

class TextUtils():
    PUNCTUATION = '[“+”+\\\\+'+ string.punctuation + ']'
    DIGIT_TO_WORD = dict((
        ('0', 'không'),
        ('1', 'một'),
        ('2', 'hai'),
        ('3', 'ba'),
        ('4', 'bốn'),
        ('5', 'năm'),
        ('6', 'sáu'),
        ('7', 'bảy'),
        ('8', 'tám'),
        ('9', 'chín'),
        ('10', 'mười'),
        ('_1' ,'mốt'),
        ('0_','lẻ')
    ))

    RANK = dict((
        (0,''),
        (1, 'nghìn'),
        (2, 'triệu'),
        (3, 'tỷ'),
        (4, 'nghìn tỷ'),
        (5, 'triệu tỷ'),
        (6,'tỷ tỷ'),
        (7,'nghìn tỷ tỷ')
    ))

    RANK_3_DIGITS = dict((
        (0,'trăm'),
        (1, 'mươi'),
        (2,'')
    ))


    @classmethod
    def normalize(cls,text:str):
        text = text.lower()
        text = cls.remove_punc(text=text)
        text = text.replace('\n',' ')
        text = text.strip(' ')
        digits = re.findall(pattern=r'\d+',string=text)
        for digit in digits:
            text = text.replace(digit,cls.convert_d_to_t(number=digit),1)
        text = re.sub(pattern=r'\s+',string=text,repl=' ')
        return text

    @classmethod
    def remove_punc(cls,text):
        return re.sub(pattern=cls.PUNCTUATION, repl='', string=text)

    @classmethod
    def convert_d_to_t(cls,number:int|float|str):
        str_number = str(number)
        less = len(str_number) % 3
        less = 3 - less if less !=0 else 0
        paded_str_number = '0'*less + str_number
        sub_number_list = [paded_str_number[i-3:i] for i in range(len(paded_str_number),0,-3)]
        text_list = []
        for i,sub in enumerate(sub_number_list):
            if i == len(sub_number_list) -1:
                text_list.append(cls.convert_3d2t(sub,rank=i,is_top=True))
                continue
            text_list.append(cls.convert_3d2t(sub, rank=i, is_top=False))
        return ' '.join(reversed(text_list))

    @classmethod
    def convert_3d2t(cls,three_digits:str,rank:int=0,is_top=False):
        if is_top:
            three_digits = three_digits.lstrip('0')
        text = str()
        for i, d in enumerate(three_digits,start=3-len(three_digits)):
            if i == 1:
                if d == '1':
                    text = text + ' ' + cls.DIGIT_TO_WORD['10']
                    continue
                elif d == '0':
                    if three_digits[i+1] == '0':
                        break
                    text = text + ' ' + cls.DIGIT_TO_WORD['0_']
                    continue
            elif i == 2:
                try:
                    if d == '1':
                        if three_digits[i-1] != '1' and three_digits[i-1] != '0':
                            text = text + ' ' + cls.DIGIT_TO_WORD['_1']
                            continue
                    elif d == '0':
                        continue
                except IndexError:
                    pass
            text = text + ' ' + cls.DIGIT_TO_WORD[d] + ' ' + cls.RANK_3_DIGITS[i]
        text = text + ' ' + cls.RANK[rank]
        return text

class Tokenizer():

    UNK = '<UNKNOWN>'
    PAD = '<PAD>'
    BLANK = '<BLANK>'
    FRACTION = 0 # -> character level
    OPTIMAL_LENGTH = 200
    OPTION_PAD = 'post'

    def __init__(self,**kwargs):
        if 'transcript_path' not in kwargs:
            self._word_counter = Counter()
        else:
            with open(file=kwargs['transcript_path'],mode='r') as f:
                contents = f.read()
            self.word_counter = Counter('_'+word for word in TextUtils.normalize(contents).split(' '))
        self._pairs_counter = defaultdict(int)
        self._merged_vocab = OrderedDict()
        self._vocab = list()
        self._is_pair = False
        self._is_split = False
        self._subword_to_index = OrderedDict()
        self._index_to_subword = OrderedDict()
        self.OPTIMAL_LENGTH = kwargs['max_len'] if 'max_len' in kwargs else self.OPTIMAL_LENGTH
        self.OPTION_PAD = kwargs['max_len'] if 'max_len' in kwargs else self.OPTIMAL_LENGTH

    def update_base_sentence(self,sentence:str):
        self._word_counter.update('_'+word for word in TextUtils.normalize(sentence).split(' '))

    def _fill_splits_counter(self):
        self._split_counter = {' '.join(word):freq for word,freq in self._word_counter.items()}
        self._fill_vocab()
        self._is_split = True

    def _fill_vocab(self):
        self._vocab.extend((self.BLANK, self.UNK, self.PAD))
        for splits in self._split_counter.keys():
            self._vocab.extend(''.join(splits))
        self._vocab = list(OrderedDict.fromkeys(self._vocab))
        self._vocab.remove(' ')
        self._vocab.sort()

    def _fill_sub_to_index(self):
        for i,v in enumerate(self._vocab):
            self._subword_to_index[v] = i
            self._index_to_subword[i] = v
        blank_index = self.get_index(sub_word=self.BLANK)
        sub_word = self.get_sub_word(index=0)
        self._subword_to_index[self.BLANK] = 0
        self._subword_to_index[sub_word] = blank_index
        self._index_to_subword[0] = self.BLANK
        self._index_to_subword[blank_index] = sub_word

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def vocab(self):
        return self._vocab

    @property
    def dictionary(self):
        return self._subword_to_index

    def _fill_pairs(self):
        v_pair_out = defaultdict(int)
        for word,freq in self._split_counter.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                v_pair_out[(symbols[i],symbols[i+1])] += freq
        self._pairs_counter = v_pair_out
        self._is_pair = True

    def _merge_vocab(self,pair:Tuple[chr]):
        bigram = ' '.join(pair)
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        v_out = defaultdict(int)
        self._merged_vocab[pair] = ''.join(pair)
        for split_w,freq in self._split_counter.items():
            sub_split = p.sub(repl=''.join(pair),string=split_w)
            v_out[sub_split] = freq
        self._split_counter = v_out

    def learn_bpe(self):
        self._fill_splits_counter()
        self._fill_pairs()
        number_of_merge = int(len(self._word_counter)*self.FRACTION)
        for _ in range(number_of_merge):
            best_pair = max(self._pairs_counter,key=self._pairs_counter.get)
            self._vocab.append(''.join(best_pair))
            self._merge_vocab(pair=best_pair)
            self._fill_pairs()
        self._fill_sub_to_index()

    def _pad_truc_sentence(self,sequence:List[str]) -> any:
        if len(sequence) >= self.OPTIMAL_LENGTH:
            return None
        else:
            padding = [self.PAD]*(self.OPTIMAL_LENGTH - len(sequence))
            if self.OPTION_PAD == 'post':
                return padding + sequence
            else:
                return sequence + padding

    def get_index(self,sub_word:str):
        assert isinstance(sub_word,str),f"'sub_word' expect string, but got {type(sub_word)}"
        return self._subword_to_index.get(sub_word,self._subword_to_index[self.UNK])

    def tokenize(self,sentence:str):
        normalized_sentence = TextUtils.normalize(sentence)
        words = normalized_sentence.split(' ')
        splits = [[c for c in '_'+word] for word in words]
        for pair,merge in self._merged_vocab.items():
            for idx,split in enumerate(splits):
                i = 0
                while i < len(split)-1:
                    if split[i] == pair[0] and split[i+1] == pair[1]:
                        split = split[:i] + [merge] + split[i+2:]
                        i = 0
                    else:
                        i += 1
                splits[idx] = split
        return splits

    def _insert_blank(self,sequence:List[str]):
        new_length = len(sequence)*2 + 1
        new_sequence = [self.BLANK]*new_length
        new_sequence[1::2] = sequence
        return new_sequence

    def encode_as_token(self,sentence:str) -> any:
        tokenized_sentence = self.tokenize(sentence)
        long_sub_word = []
        for tokenized_word in tokenized_sentence:
            long_sub_word.extend(tokenized_word)
        pad_trunc_splits = self._pad_truc_sentence(long_sub_word)
        #inserted_blank = self._insert_blank(pad_trunc_splits)
        return pad_trunc_splits

    def encode_as_index(self,sentence:str) -> any:
        long_sub_word  = self.encode_as_token(sentence=sentence)
        if long_sub_word is None:
            return
        long_sub_index = [self.get_index(sw) for sw in long_sub_word]
        return long_sub_index

    def batch_encode_as_token(self,sentences:List[str]) -> List[List[str]]:
        assert isinstance(sentences,list), TypeError('"sentences" should be list of string, use "encode_as_token" when "sentences" is string')
        batch_long_sub_word = []
        for sentence in tqdm(sentences):
            batch_long_sub_word.append(self.encode_as_token(sentence=sentence))
        return batch_long_sub_word

    def batch_encode_as_index(self,sentences:List[str]) -> List[List[int]]:
        assert isinstance(sentences,list), TypeError('"sentences" should be list of string, use "encode_as_index" when "sentences" is string')
        batch_long_index = []
        for sentence in tqdm(sentences):
            batch_long_index.append(self.encode_as_index(sentence=sentence))
        return batch_long_index

    def get_sub_word(self,index:int):
        #assert isinstance(index,int),f"'index' expect int, but got {type(index)}"
        return self._index_to_subword.get(index,self.UNK)

    def detokenize(self,sequence:List[int]) -> List[str]:
        return [self.get_sub_word(index) for index in sequence]

    def decode_token(self,sequence:List[str]) -> str:
        sentence = ''.join(sequence)
        return sentence.replace('_',' ')

    def decode_index(self,sequence:List[int]) -> str:
        decoded_token = self.detokenize(sequence=sequence)
        return self.decode_token(sequence=decoded_token)

    def batch_decode_token(self, sequences: List[List[str]]):
        assert isinstance(sequences, list), TypeError(
            '"sequences" should be list of sequence of str, use "decode_token" when "sequences" is string')
        batch_long_sub_word = []
        for sequence in tqdm(sequences):
            batch_long_sub_word.append(self.decode_token(sequence=sequence))
        return batch_long_sub_word

    def batch_decode_index(self, sequences: List[List[int]]):
        assert isinstance(sequences, list), TypeError(
            '"sequences" should be list of sequence of int, use "decode_index" when "sequences" is string')
        batch_long_token = []
        for sequence in tqdm(sequences):
            batch_long_token.append(self.decode_index(sequence=sequence))
        return batch_long_token