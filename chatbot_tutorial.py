# from __future__을 쓰는 이유는 절대 임포트가 가능하기 때문 
from __future__ import absolute_import  
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script,trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

use_cuda=torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "epu")
#print(device) # cuda check

# 사용할 cornell movie-dialogs corpus는 영화 캐릭터 대화에 관련된 데이터셋
#220,579 conversational exchanges between 10,292 pairs of movie characters(10,292쌍의 영화캐릭터 간의 220,579대화 교환)
#9,035 characters from 617 movies(617편의 영화에서 9,035자)
#304,713 total utterances(총 발화 304,713)

# 데이터의 일부 라인을 사펴보고 원래 형식을 보는것 
corpus_name="cornell movie-dialogs corpus"
corpus=os.path.join("/home/sy/Desktop",corpus_name) # 경로설정
print(corpus)
def printLines(file,n=10): # 10개의 예시 데이터 보임
    with open(file,'rb') as datafile:
        lines=datafile.readlines()
    for line in lines[:n]:
        print(line)

printLines(os.path.join(corpus,"movie_lines.txt"))

# For convenience, we'll create a nicely formatted data file in which each line contains a tab-seperated query sentence and a response sentence pair

### Create formatted data file ###
# splits each line of the file into a dictionary of fields(loadLines파일의 각 줄을 필드사전으로 분할)
# loadLines : splits each line of the file into a dictionary of fields(lineID,characterID,movieID,character,text)
# loadConversations : groups fields of lines from loadLines into conversations based on movie_conversations.txt
# extractSentencePairs : extracts pairs of sentences from conversations
def loadLines(fileName,fields):
    lines={} # dict
    with open(fileName,'r',encoding='iso-8859-1') as f:
        for line in f:
            values=line.split(" +++$+++ ")
            #extract fields
            line0bj={}
            for i,field in enumerate(fields):
                line0bj[field]=values[i]
            lines[line0bj['lineID']]=line0bj
    return lines

# Groups fields of lines from 'loadLines' into conversations based on *movie_conversations.txt*
# loadLines에 따라 라인 필드를 대화로 그룹화 
def loadConversations(fileName,lines,fields):
    conversations=[]
    with open(fileName,'r',encoding='iso-8859-1') as f: # ASCII는 7비트 인코딩 , 여기에 8비트 인코딩을 추가하며 더 많은 문자들의 표현이 가능
        for line in f:
            values=line.split(" +++$+++ ")
            #extract fields
            conv0bj={}
            for i,field in enumerate(fields):
                conv0bj[field]=values[i]
            #convert string to list (conv0bj["utteranceIDs"]=="['L598485','L598486', ...]")
            utterance_id_pattern=re.compile('L[0-9]+')
            lineIds=utterance_id_pattern.findall(conv0bj["utteranceIDs"])
            #reassemble lines
            conv0bj["lines"] = []
            for lineId in lineIds:
                conv0bj["lines"].append(lines[lineId])
            conversations.append(conv0bj)
    return conversations

# Extracts pairs of sentences from conversations(대화에서 문장 쌍을 추출)
def extractSentencePairs(conversations):
    qa_pairs=[]
    for conversation in conversations:
        # iterate over all the lines of the conversation
        for i in range(len(conversation["lines"])-1): # ignore the last line(no answer for it)
            inputLine=conversation["lines"][i]["text"].strip()
            targetLine=conversation["lines"][i+1]["text"].strip()
            #filtering wrong samples(if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine,targetLine])
    return qa_pairs
    
    
# define path to new file
datafile=os.path.join(corpus,"formatted_move_lines.txt")

delimiter='\t' # tab을 기준으로 구분 
# unescape the delimiter 
delimiter=str(codecs.decode(delimiter,"unicode_escape")) #unicode 문자열로 인코딩

# initialize lines dict,conversation list, and field ids
lines={}
conversations=[]
MOVIE_LINES_FIELDS=["lineID","characterID","movieID","character","text"]
MOVIE_CONVERSATIONS_FIELDS=["character1ID", "character2ID", "movieID", "utteranceIDs"]

# load lines and process conversations(위에서 정의한 함수 사용)
print("\nProcessing corpus...")
lines=loadLines(os.path.join(corpus,"movie_lines.txt"),MOVIE_LINES_FIELDS)
print("\nLoading conversations...")
#conversations=loadConversations(os.path.join(corpus,"movie_conversations.txt"),lines,movie_conversations_fields) 

conve=os.path.join(corpus,"movie_conversations.txt")
print(conve)

conversations=loadConversations(conve,lines,MOVIE_CONVERSATIONS_FIELDS)            

# write new csv file
print('\nWritting newly formatted file...')
with open(datafile,'w',encoding='utf-8') as outputfile:
    writer=csv.writer(outputfile,delimiter=delimiter,lineterminator='\n')
    for pair in extractSentencePairs(conversations):
        writer.writerow(pair)
# print a sample of lines
print('\nSample lines from file:')
printLines(datafile)

### load and trim data ###
# dealing with sequence of words (must create one by mapping each unique word that we encounter in our dataset to an index value)
# 어휘를 작성하고, 쿼리/응답 문장 쌍을 메모리에 로드하는 과정
# 이산 숫자 공간에 대한 explicit mapping이 없는 일련의 단어를 다루고 있다. 데이터 세트에서 만나는 각각의 고유한 단어를 인덱스값에 mapping하여 하나로 만들어야함
# default word tokens
pad_token=0 # used for padding short sentence
sos_token=1 # start-of-sentence token
eos_token=2 # end-of-sentence token

# define Voc class, which keeps a mapping from words to indexes, a reverse mapping of indexes to words,
# a count of each word and a total word count. (단어에서 인덱스로의 매핑,인덱스에서 단어로의 역방향 매핑, 각 단어의 수 및 총 단어 수를 유지함)
# The class provides methods for adding a word to the vocabulary(addWord), adding all words in a sentence(addSentence)
# and trimming inffrequently seen words(trim). 

class Voc:
    def __init__(self,name):
        self.name=name
        self.trimmed=False
        self.word2index={}
        self.word2count={}
        self.index2word={pad_token:"PAD",sos_token:"SOS",eos_token:"EOS"}
        self.num_words=3 # count sos,eos,pad
    
    def addSentence(self,sentence): # 문장에 모든 단어를 추가 
        for word in sentence.split(' '):
            self.addWord(word)
    
    def addWord(self,word):  # 단어를 어휘에 추가 
        if word not in self.word2index:
            self.word2index[word]=self.num_words
            self.word2count[word]=1
            self.index2word[self.num_words]=word
            self.num_words +=1
        else:
            self.word2count[word]+=1
    
    #remove words below a certain count threshold
    def trim(self,min_count):  # 최소 단어 수보다 작으면 단어 자르기 (드물게 보이는 단어를 자르기 위한 메소드 제공)
        if self.trimmed:
            return
        self.trimmed=True
        
        keep_words=[]
        
        for k,v in self.word2count.items():
            if v>=min_count:
                keep_words.append(k)
                
        print('keep_words {}/{} ={:.4f}'.format(len(keep_words),len(self.word2index),len(keep_words)/len(self.word2index)))
        
        #reinitialize dictionaries
        self.word2index={}
        self.word2count={}
        self.index2word={pad_token:"PAD",sos_token:"SOS",eos_token:"EOS"}
        self.num_words=3 #count default tokens
        
        for word in keep_words:
            self.addWord(word)   
            
# 이제 어휘와 쿼리/응답 문장 쌍을 조합할 수 있다. 이 데이터를 사용하기 전에 사전 처리를 수행해야 한다. 
# 1. first, convert the unicode strings to ASCII using (unicodeToAscii)
# 2. second, convert all letters to lowercase and trim all non-letter characters except for basic punctuation(normalizeString)
# 3. finally, to aid in training convergence, we will filter out sentences with length greater than th MAX_LENGTH threshold(filterPairs)
"""유니코드를 아스키코드로 변환하고, 모든 문자를 소문자로 변환하고 기본 구두점(normalizeString)을 제외한 모든 문자 이외의 문자를 자름
max_length임계값보다 길이가 긴 문장이면 필터링(삭제)- 여기선 10으로 지정"""

MAX_LENGTH=10 # maximum sentence length to consider 
# turn a unicode string to plain ASCII 
def unicodeToAscii(s):
    return ''.join(
    c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c)!='Mn'
    )

#Lowercase,trim,and remove non-letter characters(소문자화, 문자가 아닌것 제거, 공백 처리)
def normalizeString(s): 
    s=unicodeToAscii(s.lower().strip())
    s=re.sub(r"([.!?])",r" \1",s)
    s=re.sub(r"[^a-zA-Z.!?]+",r" ",s)
    s=re.sub(r"\s+",r" ",s).strip()
    return s

# read query/response pairs and return a voc object
def readVocs(datafile,corpus_name):
    print("Reading lines...")
    # read the file and split into lines
    lines=open(datafile,encoding='utf-8').\
        read().strip().split('\n')
    # split every line into pairs and normalize
    pairs=[[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc=Voc(corpus_name)
    return voc,pairs

# return True if both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    #input sequences need to preserve the last word for eos token
    return len(p[0].split(' '))<MAX_LENGTH and len(p[1].split(' '))<MAX_LENGTH

# filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# using the functions defined above, return a polulated voc object and pairs list
def loadPrepareData(corpus,corpus_name,datafile,save_dir):
    print("start preparing training data...")
    voc,pairs=readVocs(datafile,corpus_name)
    print("read {!s} sentence pairs".format(len(pairs)))
    pairs=filterPairs(pairs)
    print("trimmed to {!s} sentence pairs".format(len(pairs)))
    print("counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("counted words:",voc.num_words)
    return voc,pairs

# load/assemble voc and pairs
save_dir=os.path.join("/home/sy/Desktop","save")
voc,pairs=loadPrepareData(corpus,corpus_name,datafile,save_dir)
# print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)
    
# 1. trim words used under MIN_COUNT threshold using the voc.trim function(함수를 이용하여 min_count임계값 미만으로 사용되는 단어 자르기:voc.trim)
# 2. filter out pairs with trimmed words(잘려진 단어로 쌍을 필터링)
MIN_COUNT=3 # minimum word count threshold for trimming

def trimRareWords(voc,pairs,MIN_COUNT):
    # trim words used under the MIN_COUNT from the voc 
    voc.trim(MIN_COUNT)
    # filter out pairs with trimmed words
    keep_pairs=[]
    for pair in pairs:
        input_sentence=pair[0]
        output_sentence=pair[1]
        keep_input=True
        keep_output=True
        # check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input=False
                break
        
        # check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output=False
                break
        
        # only keep pairs that do not contain trimmed words in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)
            
    print("trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs),len(keep_pairs),len(keep_pairs)/len(pairs)))
    return keep_pairs

#trim voc and pairs
pairs=trimRareWords(voc,pairs,MIN_COUNT)

### prepare data for models ###
# In that tutorial, we use a batch size of 1, meaning that all we have to do is convert the words 
# in our sentence pairs to their corresponding indexes from the vocabulary and feed this to the models.
# However, if you’re interested in speeding up training and/or would like to leverage GPU parallelization capabilities, 
# you will need to train with mini-batches.
"""
단어를 색인 및 제로 패드로 변환하여 영어문장을 텐서로 간단히 변환하면 텐서는 모양(batch_size,max_length)을 가지며 첫번째 차원을 
색인화하면 모든 시간단계에서 전체 sequence를 반환
그러나 time과 batch의 모든 시퀀스에서 배치를 색인화할 수 있어야 한다. 따라서 입력을 일괄적으로 (max_length,batch_size)로 바구면서 첫번째 
차원에 대한 색인의 생성은 일괄 처리의 모든 문장에서 시간 단계를 반환 -> 이 transpose를 zeroPadding함수에서 처리
"""

def indexesFromSentence(voc,sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [eos_token]
    
def zeroPadding(l,fillvalue=pad_token):  # sentences shorter than MAX_LENGTH are zero-padding
    return list(itertools.zip_longest(*l,fillvalue=fillvalue))
    
def binaryMatrix(l,value=pad_token):
    m=[]
    for i,seq in enumerate(l):
        m.append([])
        for token in seq:
            if token==pad_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m
    
 #returns padded input sequence tensor and lengths
# 문장을 텐서로 변환하는 프로세스를 처리하여 궁극적으로 올바른 모양의 zero padding tensor를 만든다. 
# lengths 배치의 각 시퀀스에 대한 텐서를 반환하여 나중에 디코더로 전달 
def inputVar(l,voc):
    indexes_batch=[indexesFromSentence(voc,sentence) for sentence in l]
    lengths=torch.tensor([len(indexes) for indexes in indexes_batch])
    padList=zeroPadding(indexes_batch)
    padVar=torch.LongTensor(padList)
    return padVar, lengths

#returns padded target sequence tensor,padding mask,and max target length
# lengths 텐서를 반환하는 대신 binarymaskTensor와 최대 문장 길이를 반환 
# binarymaskTensor는 출력 대상인 텐서와 모양은 동일하지만 pad_token인 모든 요소는 False이고, 나머지는 1이다. 
def outputVar(l,voc):
    indexes_batch=[indexesFromSentence(voc,sentence) for sentence in l]
    max_target_len=max([len(indexes) for indexes in indexes_batch])
    padList=zeroPadding(indexes_batch)
    mask=binaryMatrix(padList)
    mask=torch.BoolTensor(mask) #dtype=torch.bool 
    padVar=torch.LongTensor(padList)
    return padVar,mask,max_target_len

#return all items for a given batch of pairs
def batch2TrainData(voc,pair_batch): # pair과 pair을 취하고 위에서 언급한 함수를 이용해 입력 및 대상 텐서를 반환
    pair_batch.sort(key=lambda x: len(x[0].split(" ")),reverse=True)
    input_batch,output_batch= [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp,lengths=inputVar(input_batch,voc)
    output,mask,max_target_len=outputVar(output_batch,voc)
    return inp,lengths,output,mask,max_target_len

# example for validation
small_batch_size=5
batches=batch2TrainData(voc,[random.choice(pairs) for _ in range(small_batch_size)])
input_variable,lengths,target_variable,mask,max_target_len=batches

print("input_variable:",input_variable)
print("lengths:",lengths)
print("target_variable:",target_variable)
print("mask:",mask)
print("max_target_len:",max_target_len)

### Define Model(seq2seq model) ###
# chatbot's brain is sequence2sequence model 
# 이 모델의 목표는 가변(변할수있는) 길이 시퀀스를 입력으로 사용하고 고정 길이 모델을 사용하여 가변 길이 시퀀스를 출력으로 반환하는 것
# 두개의 분리된 반복 신경망을 함께 사용하면 이 작업을 수행할 수 있다. 
# 하나의 RNN(encoder)은 가변 길이 입력 시퀀스를 고정 길이 context 벡터로 인코딩하는 인코더 역할을 함
# 이론적으로 context 벡터(RNN의 최종 숨겨진 계층)에는 bot에 입력되는 쿼리 문장에 대한 의미 정보가 포함된다. 
# 두번째 RNN(decoder)에서 입력단어와 context 벡터를 가져와, 시퀀스의 다음 단어에 대한 추측과 다음 반복에 사용할 hidden state를 반환

# computation graph
# 1. convert word indexes to embeddings
# 2. pack padded batch of sequences for RNN module
# 3. forward pass through GRU
# 4. unpack padding(using nn.utils.rnn.pack_padded_sequence and nn.utils.rnn.pad_packed_sequence)
# 5. sum bidirectional GRU outputs
# 6. return output and final hidden state

# torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True) # default
# torch.nn.utils.rnn.pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None) # default

"""<inputs>
input_seq : batch of input sentences; shape=(max_length,batch_size)
 -> 입력문장의 batch
input_lengths: list of sentence lengths corresponding to each sentence in the batch ; shpae=(batch_size)
 -> 배치의 각 문장에 해당하는 문장 길이의 목록
hidden : hidden state ; shape=(n_layers xnum_directions,batch_size,hidden_size)

<outputs>
outputs : output features from the last hidden layer of the GRU(sum of bidirectional outputs); shape=(max_length,batch_size,hidden_size)
 -> GRU의 마지막 hidden_state의 출력 기능
hidden : updated hidden state from GRU ; shape=(n layer xnum_directions,batch_size,hidden_size)
 -> GRU에서 hidden state를 업데이트
"""

# Encoder
class EncoderRNN(nn.Module):
    def __init__(self,hidden_size,embedding,n_layers=1,dropout=0):   
        super(EncoderRNN,self).__init__()
        self.n_layers=n_layers
        self.hidden_size=hidden_size
        self.embedding=embedding
        # initailize GRU : the input_size and hidden_size parameters are both set to 'hidden_size'
        # because our input_size is a word embedding with number of features == hidden_size
        # GRU의 양방향변형(bidirectional)을 사용 - 본질적으로 두 개의 독립적인 RNN이 있다는 것을 의미 
        # 하나는 정상적인 순서로 입력 시퀀스를 공급하고, 다른 하나는 입력 순서를 역순으로 공급 
        # 각 네트워크의 출력은 각 시간 단계에서 합산됨 
        # bidirectional GRU를 이용하면 과거와 미래의 context를 모두 인코딩할 수 있는 장점이 있다. 
        self.gru=nn.GRU(hidden_size,hidden_size,n_layers,dropout=(0 if n_layers==1 else dropout),bidirectional=True)
    
    def forward(self,input_seq,input_lengths,hidden=None):
        #convert word indexes to embeddings
        # embedding층이 임의의 크기의 기능 공간에서 word index를 인코딩하는데 사용된다. 
        # 이 model안에서, layer는 hidden_size의 feature space에 각 단어를 mapping해야한다. 
        embedded=self.embedding(input_seq)
        #pack paddded batch of sequence for RNNmodule
        packed=nn.utils.rnn.pack_padded_sequence(embedded,input_lengths)
        #forward pass through GRU
        outputs,hidden=self.gru(packed,hidden)
        #unpack padding
        outputs,_=nn.utils.rnn.pad_packed_sequence(outputs)
        #sum bidirectional GRU outputs
        outputs=outputs[:,:,:self.hidden_size]+outputs[:,:,self.hidden_size:]
        return outputs,hidden
  
# decoderRNN은 토큰마다 응답문장을 생성
# Luong attention layer (paper참고)
class Attn(nn.Module):
    def __init__(self,method,hidden_size):
        super(Attn,self).__init__()
        self.method=method
        if self.method not in ['dot','general','concat']:
            raise ValueError(self.method,"is not an appropriate attention method.")
        self.hidden_size=hidden_size
        if self.method=='general':
            self.attn=nn.Linear(self.hidden_size,hidden_size)
        elif self.method=='concat':
            self.attn=nn.Linear(self.hidden_size*2,hidden_size)
            self.v=nn.Parameter(torch.FloatTensor(hidden_size))
     
    #three different alternatives
    def dot_score(self,hidden,encoder_output):
        return torch.sum(hidden*encoder_output,dim=2)
    
    def general_score(self,hidden,encoder_output):
        energy=self.attn(torch.cat((hidden.expand(encoder_output.size(0),-1,-1),encoder_output),2)).tanh()
        return torch.sum(self.v*energy,dim=2)
    
    def concat_score(self,hidden,encoder_output):
        energy=self.attn(torch.cat((hidden.expand(encoder_output.size(0),-1,-1),encoder_output),2)).tanh()
        return torch.sum(self.v*enerhy,dim=2)
    
    def forward(self,hidden,encoder_outputs):
        # calculate the attention weights(energies) based on the given method
        if self.method=='general':
            attn_energies=self.general_score(hidden,encoder_outputs)
        elif self.method=='concat':
            attn_energies=self.concat_score(hidden,encoder_outputs)
        elif self.method=='dot':
            attn_energies=self.dot_score(hidden,encoder_outputs)
        
        #transpose max_length and batch_size dimensions
        attn_energies=attn_energies.t() # transpose
        
        #return the softmax normalized probability score (with added dimension) , shape = (batch_size,1,max_length)
        return F.softmax(attn_energies,dim=1).unsqueeze(1) 

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self,attn_model,embedding,hidden_size,output_size,n_layers=1,dropout=0.1):
        super(LuongAttnDecoderRNN,self).__init__()
        # keep for reference
        self.attn_model=attn_model
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.n_layers=n_layers
        self.dropout=dropout
        
        # define layers
        self.embedding=embedding
        self.embedding_dropout=nn.Dropout(dropout)
        self.gru=nn.GRU(hidden_size,hidden_size,n_layers,dropout=(0 if n_layers==1 else dropout))
        self.concat=nn.Linear(hidden_size*2,hidden_size)
        self.out=nn.Linear(hidden_size,output_size)
        self.attn=Attn(attn_model,hidden_size)
        
    def forward(self,input_step,last_hidden,encoder_outputs):
        # note : we run this one step(word) at a time
        # get embedding of current input word
        embedded=self.embedding(input_step)
        embedded=self.embedding_dropout(embedded)
        # forward through unidirectional GRU
        rnn_output,hidden=self.gru(embedded,last_hidden)
        # calculate attention weights from the current GRU output
        attn_weights=self.attn(rnn_output,encoder_outputs)
        # multiply attention weights to encoder outputs to get new "weighted sum" context vector 
        context=attn_weights.bmm(encoder_outputs.transpose(0,1))
        # concatenate weighted context vector and GRU output using Luong eq.5
        rnn_output=rnn_output.squeeze(0)
        context=context.squeeze(1)
        concat_input=torch.cat((rnn_output,context),1)
        concat_output=torch.tanh(self.concat(concat_input))
        
        #predict next word using Luong eq.6
        output=self.out(concat_output)
        output=F.softmax(output,dim=1)
        #return output and final hidden state
        return output,hidden        
        
 ### Define training precedure ###
# masked loss : to calculate our loss based on our decoder's output tensor, the target tensor, and a binary mask tensor
# this loss function calculates the average negative log likelihood of the elements that correspond to a 1 in the mask tensor
def maskNLLLoss(inp,target,mask):
    nTotal=mask.sum()
    crossEntropy=-torch.log(torch.gather(inp,1,target.view(-1,1)).squeeze(1))
    loss=crossEntropy.masked_select(mask).mean()
    loss=loss.to(device)
    return loss,nTotal.item()
    
"""
<sequence of operations>
1. forward pass entire input batch through encoder
2. initialize decoder inputs as sos_token, and hidden state as the encoder's final hidden state
3. forward input batch sequence through decoder one time step at a time
4. if teacher forcing: set next decoder input as the current target
   else : set next decoder input as current decoder output
5. calculate and accumulate loss
6. perform backpropagation
7. clip gradients
8. update encoder and decoder model parameters
"""

# train
def train(input_variable,lengths,target_variable,mask,max_target_len,
          encoder,decoder,embedding,encoder_optimizer,decoder_optimizer,batch_size,clip,max_length=MAX_LENGTH):
    #zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    #set device options
    input_variable=input_variable.to(device)
    lengths=lengths.to(device)
    target_variable=target_variable.to(device)
    mask=mask.to(device)
    
    # initialize variable
    loss=0
    print_losses=[]
    n_totals=0
    
    # forward pass through encoder
    encoder_outputs,encoder_hidden=encoder(input_variable,lengths)
    
    # create initial decoder input (start with sos token for each sentence)
    decoder_input=torch.LongTensor([[sos_token for _ in range(batch_size)]])
    decoder_input=decoder_input.to(device)
    
    # set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden=encoder_hidden[:decoder.n_layers]
    
    # determine if we are using teacher forcing this iteration 
    use_teacher_forcing=True if random.random()<teacher_forcing_ratio else False
    
    # forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output,decoder_hidden=decoder(decoder_input,decoder_hidden,encoder_outputs)
            #teacher forcing : next input is current target
            decoder_input=target_variable[t].view(1,-1)
            # calculate and accumulate loss
            mask_loss,nTotal=maskNLLLoss(decoder_output,target_variable[t],mask[t])
            loss+=mask_loss
            print_losses.append(mask_loss.item()*nTotal)
            n_totals+=nTotal
    else:
        for t in range(max_target_len):
            decoder_output,decoder_hidden=decoder(decoder_input,decoder_hidden,encoder_outputs)
            # no teacher forcing : next input is decoder's own current output
            _,topi=decoder_output.topk(1)
            decoder_input=torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input=decoder_input.to(device)
            # calculate and accumulate loss
            mask_loss,nTotal=maskNLLLoss(decoder_output,target_variable[t],mask[t])
            loss+=mask_loss
            print_losses.append(mask_loss.item()*nTotal)
            n_totals+=nTotal
            
    # perform backpropagation 
    loss.backward()
    
    # clip gradients : gradients are modified in place 
    _=nn.utils.clip_grad_norm_(encoder.parameters(),clip)
    _=nn.utils.clip_grad_norm_(decoder.parameters(),clip)
    
    # adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return sum(print_losses)/n_totals    
    
    
def trainIters(model_name,voc,pairs,encoder,decoder,encoder_optimizer,decoder_optimizer,
              embedding,encoder_n_layers,decoder_n_layers,save_dir,n_iteration,batch_size,print_every,
              save_every,clip,corpus_name,loadFilename):
    #load batches for each iteration
    training_batches=[batch2TrainData(voc,[random.choice(pairs) for _ in range(batch_size)]) for _ in range(n_iteration)]
    #initializations
    print("initializing...")
    start_iteration=1
    print_loss=0
    if loadFilename:
        start_iteration=checkpoint['iteration']+1
        
    #training loop
    print("trainig...")
    for iteration in range(start_iteration,n_iteration+1):
        training_batch=training_batches[iteration-1]
        # extract fields from batch
        input_variable,lengths,target_variable,mask,max_target_len=training_batch
        
        #run a training iteration with batch
        loss=train(input_variable,lengths,target_variable,mask,max_target_len,encoder,decoder,embedding,
                  encoder_optimizer,decoder_optimizer,batch_size,clip)
        print_loss+=loss
        # print progress
        if iteration%print_every==0:
            print_loss_avg=print_loss/print_every
            print("iteration : {}; percent complete:{:.1f}%; average loss:{:.4f}".format(iteration,iteration/n_iteration*100,print_loss_avg))
            print_loss=0
        # save checkpoint
        if(iteration%save_every==0):
            directory=os.path.join(save_dir,model_name,corpus_name,'{}-{}_{}'.format(encoder_n_layers,decoder_n_layers,hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration':iteration,
                'en':encoder.state_dict(),
                'de':decoder.state_dict(),
                'en_opt':encoder_optimizer.state_dict(),
                'de_opt':decoder_optimizer.state_dict(),
                'loss':loss,
                'voc_dict':voc.__dict__,
                'embedding':embedding.state_dict()
                
            },os.path.join(directory,'{}_{}.tar'.format(iteration,'checkpoint')))
            
### Define Evaluation ###

class GreedySearchDecoder(nn.Module):
    def __init__(self,encoder,decoder):
        super(GreedySearchDecoder,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
    def forward(self,input_seq,input_length,max_length):
        # forward input through encoder model 
        encoder_outputs,encoder_hidden=self.encoder(input_seq,input_length)
        # prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden=encoder_hidden[:decoder.n_layers]
        #initialize decoder input with sos_token
        decoder_input=torch.ones(1,1,device=device,dtype=torch.long)*sos_token
        # initialize tensors to append decoded words to 
        all_tokens=torch.zeros([0],device=device,dtype=torch.long)
        all_scores=torch.zeros([0],device=device)
        # iteratively decode one word token at a time
        for _ in range(max_length):
            # forward pass through decoder 
            decoder_output,decoder_hidden=self.decoder(decoder_input,decoder_hidden,encoder_outputs)
            # obtain most likely word token and its softmax score
            decoder_scores,decoder_input=torch.max(decoder_output,dim=1)
            # record token and score
            all_tokens=torch.cat((all_tokens,decoder_input),dim=0)
            all_scores=torch.cat((all_scores,decoder_score),dim=0)
            # prepare current token to be next decoder input (add a dimension)
            decoder_input=torch.unsqueeze(decoder_input,0)
        # return collections of word tokens and scores
        return all_tokens,all_scores
        
# Evaluate function manages the low-level process of handling the input sentence.
# first, format the sentence as an input batch of word indexes with batch_size==1
# 이 tutorial의 경우, batch_size가 1이기 때문에 lengths의 값은 scalar값이다. 왜냐면 한번에 한문장씩 평가하기 때문

def evaluate(encoder,decoder,voc,sentence,max_length=MAX_LENGTH):
    ## format input sentence as a batch
    # words -> inputs
    indexes_batch=[indexesFromSentence(voc,sentence)]
    # create lengths tensor
    lengths=torch.tensor([len(indexes) for indexes in indexes_batch])
    # transpose dimensions of batch to match model's expectations
    input_batch=torch.LongTensor(indexes_batch).transpose(0,1)
    # use appropriate device
    input_batch=input_batch.to(device)
    lengths=lengths.to(device)
    # decode sentence with searcher
    tokens,scores=searcher(input_batch,lengths,max_length)
    # indexes -> words
    decoded_words=[voc.index2word[token.item()] for token in tokens]
    return decoded_words

def evaluateInput(encoder,decoder,searcher,voc):
    input_sentence=''
    while(1):
        try:
            # get input sentence
            input_sentence=input('>')
            # check if it is quit case
            if input_sentence =='q' or input_sentence=='quit': break
            # normalize sentence
            input_sentence=normalizeString(input_sentence)
            # evaluate sentence
            output_words=evaluate(encoder,decoder,searcher,voc,input_sentence)
            # format and print response sentence 
            output_words[:]=[x for x in output_words if not (x=='EOS' or x=='PAD')]
            print('Bot : ',' '.join(output_words))
            
        except KeyError:
            print("Error: Encountered unknown word.")
            
### Run Model ###

model_name='cb_model'
attn_model='dot'
#attn_model='general'
#attn_model='concat'
hidden_size=500
encoder_n_layers=2
decoder_n_layers=2
dropout=0.1
batch_size=64

# set checkpoint to load from ; set to None if starting from scratch
loadFilename=None
checkpoint_iter=4000

""" loadFilename=os.path.join(save_dir,model_name,corpus_name,'{}-{}_{}'.format(encoder_n_layers,decoder_n_layers,hidden_size),
    '{}_check_point.tar'.format(checkpoint_iter))"""

# load model if a loadFilename is provided
if loadFilename:
    #if loading on same machine the model was trained on
    checkpoint=torch.load(loadFilename)
    encoder_sd=checkpoint['en']
    decoder_sd=checkpoint['de']
    encoder_optimizer_sd=checkpoint['en_opt']
    decoder_optimizer_sd=checkpoint['de_opt']
    embedding_sd=checkpoint['embedding']
    voc.__dict__=checkpoint['voc_dict']
print('Building encoder and decoder')

# initialize word embeddings
embedding=nn.Embedding(voc.num_words,hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# initialize encoder&decoder model 
encoder=EncoderRNN(hidden_size,embedding,encoder_n_layers,dropout)
decoder=LuongAttnDecoderRNN(attn_model,embedding,hidden_size,voc.num_words,decoder_n_layers,dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# use appropriate device
encoder=encoder.to(device)
decoder=decoder.to(device)
print('models built and ready to go!')

### Run Training ###
# configure training/optimization
clip=50.0
teacher_forcing_ratio=1.0
learning_rate=1e-3
decoder_learning_ratio=5.0
n_iteration=50 # n_iteration으로 조절
print_every=1
save_every=500

# ensure dropout layers are in train mode
encoder.train()
decoder.train()

# initialize optimizers
print('building optimizers ...')
encoder_optimizer=optim.Adam(encoder.parameters(),lr=learning_rate)
decoder_optimizer=optim.Adam(decoder.parameters(),lr=learning_rate*decoder_learning_ratio)

if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# if you have cuda, configure cuda to call
for state in encoder_optimizer.state.values():
    for k,v in state.items():
        if isinstance(v,torch.Tensor):
            state[k]=v.cuda()

for state in decoder_optimizer.state.values():
    for k,v in state.items():
        if isinstance(v,torch.Tensor):
            state[k]=v.cuda()

# run training iterations
print('starting training!...')
trainIters(model_name,voc,pairs,encoder,decoder,encoder_optimizer,decoder_optimizer,
              embedding,encoder_n_layers,decoder_n_layers,save_dir,n_iteration,batch_size,print_every,save_every,clip,corpus_name,loadFilename)

# run evaluation
# set dropout with model, run the following block

encoder.eval()
decoder.eval()

#initialize search module
searcher=GreedySearchDecoder(encoder,decoder)

evaluateInput(encoder,decoder,searcher,voc) # 대화 실행 (q나 quit를 입력하면 대화종료)
