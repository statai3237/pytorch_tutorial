from __future__ import unicode_literals,print_function,division
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 언어의 각 단어를 one-hot vector or unit word(index of word)를 제외하고 0으로 이루어진 매우큰 vector로 표현할 것
# 언어에 존재하는 수십개의 문자와 비교할떄 더 많은 단어가 있기때문에 encoding vector는 훨씬 크다. 
# 그러나 이 tutorial에서는 trim the data to only use a few thousand words per language.

sos_token=0
eos_token=1

class Lang:
    def __init__(self,name):
        self.name=name
        self.word2index={}
        self.word2count={}
        self.index2word={0:"SOS",1:"EOS"}
        self.n_words=2 # count SOS and EOS
    
    def addSentence(self,sentence):
        for word in sentence.split(' '):
            self.addWord(word)
            
    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word]=self.n_words
            self.word2count[word]=1
            self.index2word[self.n_words]=word
            self.n_words+=1
        else:
            self.word2count[word]+=1
            
            
   # the files unicode string to plain ASCII (therefore, that will turn unicode characters to ASCII,
# make everything lowercase,and trim most punctuation.)
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD',s)   # NFD — 정규형 정준 분해(Normalization Form Canonical Decomposition).
        if unicodedata.category(c)!='Mn'
    )

# lowercase,trim,and remove non-letter characters

def normalizeString(s):
    s=unicodeToAscii(s.lower().strip())
    s=re.sub(r"([.!?])",r" \1",s)
    s=re.sub(r"[^a-zA-Z.!?]+",r" ",s)
    return s
    
"""
To read the data file we will split the file into lines,and then split lines into pairs.
The files ar all English -> Other Language, so if we want to translate from Other language->English
I added the reverse flag to reverse the pairs.
"""

def readLangs(lang1,lang2,reverse=False):
    print("Reading Lines...")
    # read the file and split into lines
    lines=open('/home/sy/Desktop/%s-%s.txt'%(lang1,lang2),encoding='utf-8').\
    read().strip().split('\n')
    
    # split every line into pairs and normalize
    pairs=[[normalizeString(s) for s in l.split('\t')] for l in lines]
    
    #reverse pairs, make language instances
    if reverse:
        pairs=[list(reversed(p)) for p in pairs]
        input_lang=Lang(lang2) 
        output_lang=Lang(lang1)
    else:
        input_lang=Lang(lang1)
        output_lang=Lang(lang2)
    return input_lang,output_lang,pairs
    

MAX_LENGTH=10

eng_prefixes=(
    "i am","i m",
    "he is","he s",
    "she is","she s",
    "you are","you re",
    "we are","we re",
    "they are","they re"
)

def filterPair(p):
    return len(p[0].split(' '))<MAX_LENGTH and \
        len(p[1].split(' '))<MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

"""
The full process for preparing the data is:
1. read text file and split into lines, split lines into pairs
2. normalize text, filter by length and content
3. make word lists from sentences in pairs
"""

def prepareData(lang1,lang2,reverse=False):
    input_lang,output_lang,pairs=readLangs(lang1,lang2,reverse)
    print("Read %s sentence pairs"%len(pairs))
    pairs=filterPairs(pairs)
    print("Trimmed to %s sentence pairs"%len(pairs))
    print("counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("counted words:")
    print(input_lang.name,input_lang.n_words)
    print(output_lang.name,output_lang.n_words)
    return input_lang,output_lang,pairs

input_lang,output_lang,pairs=prepareData('eng','fra',True)
print(random.choice(pairs))

# seq2seq network의 encoder는 입력문장에서 모든 단어에 대한 일부 값을 출력하는 RNN이다. 
# 모든 입력 단어에 대해 인코더는 vector와 hidden state를 출력하고 다음 입력단어에 hidden state를 사용한다. 

# Encoder : 입력문장에서 모든 단어에 대한 일부 값을 출력하는 RNN
class EncoderRNN(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(EncoderRNN,self).__init__()
        self.hidden_size=hidden_size
        
        self.embedding=nn.Embedding(input_size,hidden_size)
        self.gru=nn.GRU(hidden_size,hidden_size)
        
    def forward(self,input,hidden):
        embedded=self.embedding(input).view(1,1,-1)
        output=embedded
        output,hidden=self.gru(output,hidden)
        return output,hidden
    
    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size,device=device)
        

# 가장 간단한 seq2seq 디코더에서는 인코더의 마지막 출력만 사용. 이 마지막 출력은 전체 시퀀스에서 context를 인코딩할때 
# context vector라고도 함 , context vector는 디코더의 initial hidden state로 사용된다. 
# decoding의 모든 단계에서, 디코더의 input token 및 hidden state가 부여된다. 
# 초기 input token은 문자열 시작 <SOS>토큰이고 첫번째 hidden state는 context vector(인코더의 마지막 hidden state)

class DecoderRNN(nn.Module):
    def __init__(self,hidden_size,output_size):
        super(DecoderRNN,self).__init__()
        self.hidden_size=hidden_size
        
        self.embedding=nn.Embedding(output_size,hidden_size)
        self.gru=nn.GRU(hidden_size,hidden_size)
        self.out=nn.Linear(hidden_size,output_size)
        self.softmax=nn.LogSoftmax(dim=1)
        
    def forward(self,input,hidden):
        output=self.embedding(input).view(1,1,-1)
        output=F.relu(output)
        output,hidden=self.gru(output,hidden)
        output=self.softmax(self.out(output[0]))
        return output,hidden
    
    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size,device=device)
        

# Attention Decoder
# first, we calculate a set of attention weights.
# these will be multiplied by the encoder output vectors to create a weighted combination.
# the result(attn_applied) should contain information about that specific part of the input sequence, and 
# thus help the decoder choose the right output words

class AttnDecoderRNN(nn.Module):
    def __init__(self,hidden_size,output_size,dropout_p=0.1,max_length=MAX_LENGTH):
        super(AttnDecoderRNN,self).__init__()
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.dropout_p=dropout_p
        self.max_length=max_length
        
        self.embedding=nn.Embedding(self.output_size,self.hidden_size)
        self.attn=nn.Linear(self.hidden_size*2,self.max_length)
        self.attn_combine=nn.Linear(self.hidden_size*2,self.hidden_size)
        self.dropout=nn.Dropout(self.dropout_p)
        self.gru=nn.GRU(self.hidden_size,self.hidden_size)
        self.out=nn.Linear(self.hidden_size,self.output_size)
        
    def forward(self,input,hidden,encoder_outputs):
        embedded=self.embedding(input).view(1,1,-1)
        embedded=self.dropout(embedded)
        
        attn_weights=F.softmax(self.attn(torch.cat((embedded[0],hidden[0]),1)),dim=1)
        attn_applied=torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))
        
        output=torch.cat((embedded[0],attn_applied[0]),1)
        output=self.attn_combine(output).unsqueeze(0)
        
        output=F.relu(output)
        output,hidden=self.gru(output,hidden)
        
        output=F.log_softmax(self.out(output[0]),dim=1)
        return output,hidden,attn_weights
    
    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size,device=device)
  
# training (preparing training Data)
"""
To train, for each pair we will need an input tensor(indexes of the words in the input sentence) 
and target tensor(indexes of the words in the target sentence).
While creating these vectors we will append the EOS token to both sequences.

"""
def indexesFromSentence(lang,sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang,sentence):
    indexes=indexesFromSentence(lang,sentence)
    indexes.append(eos_token)
    return torch.tensor(indexes,dtype=torch.long,device=device).view(-1,1)

def tensorsFromPair(pair):
    input_tensor=tensorFromSentence(input_lang,pair[0])
    target_tensor=tensorFromSentence(output_lang,pair[1])
    return (input_tensor,target_tensor)
    
    
# Train
teacher_forcing_ratio=0.5
def train(input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion,max_length=MAX_LENGTH):
    encoder_hidden=encoder.initHidden()
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_length=input_tensor.size(0)
    target_length=target_tensor.size(0)
    
    encoder_outputs=torch.zeros(max_length,encoder.hidden_size,device=device)
    
    loss=0
    
    for ei in range(input_length):
        encoder_output,encoder_hidden=encoder(input_tensor[ei],encoder_hidden)
        encoder_outputs[ei]=encoder_output[0,0]
    
    decoder_input=torch.tensor([[sos_token]],device=device)
    decoder_hidden=encoder_hidden
    
    use_teacher_forcing=True if random.random()<teacher_forcing_ratio else False
    
    if use_teacher_forcing:
        # teacher_forcing : feed the target as the next input
        for di in range(target_length):
            decoder_output,decoder_hidden,decoder_attention=decoder(decoder_input,decoder_hidden,encoder_outputs)
            loss+=criterion(decoder_output,target_tensor[di])
            decoder_input=target_tensor[di] # teacher forcing
            
    else:
        # without teacher forcing : use its own predictions as the next input
        for di in range(target_length):
            decoder_output,decoder_hidden,decoder_attention=decoder(decoder_input,decoder_hidden,encoder_outputs)
            topv,topi=decoder_output.topk(1)
            decoder_input=topi.squeeze().detach() # detach from history as input
            loss+=criterion(decoder_output,target_tensor[di])
            if decoder_input.item()==eos_token:
                break
    loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item()/target_length

# current time과 진행률 %에서 경과시간과 남은 예상 시간을 인쇄하는 도우미 기능하는 함수 
import time
import math

def asMinutes(s):
    m=math.floor(s/60)
    s-=m*60
    return '%dm %ds' % (m,s)

def timeSince(since,percent):
    now=time.time()
    s=now-since
    es=s/(percent)
    rs=es-s
    return '%s (- %s)' %(asMinutes(s),asMinutes(rs))
    
"""
<whole training process looks like this:
start a timer
initialize optimizers and criterion
create set of training pairs
start empty losses array for plotting
"""

def trainIters(encoder,decoder,n_iters,print_every=1000,plot_every=100,learning_rate=0.01):
    start=time.time()
    plot_losses=[]
    print_loss_total=0 # reset evert print_every
    plot_loss_total=0 # reset evert plot_every
    
    encoder_optimizer=optim.SGD(encoder.parameters(),lr=learning_rate)
    decoder_optimizer=optim.SGD(decoder.parameters(),lr=learning_rate)
    training_pairs=[tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]
    
    criterion=nn.NLLLoss()
    for iter in range(1,n_iters+1):
        training_pair=training_pairs[iter-1]
        input_tensor=training_pair[0]
        target_tensor=training_pair[1]
        
        loss=train(input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion)
        print_loss_total+=loss
        plot_loss_total+=loss
        
        if iter%print_every==0:
            print_loss_avg=print_loss_total/print_every
            print_loss_total=0
            print('%s (%d %d%%) %.4f'%(timeSince(start,iter/n_iters),iter,iter/n_iters*100,print_loss_avg))
        
        if iter%plot_every==0:
            plot_loss_avg=plot_loss_total/plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total=0
    showPlot(plot_losses)

# plotting results
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig,ax=plt.subplots()
    # this locator puts ticks at regular intervals
    loc=ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    
# evaluation 
"""
every time it predicts a word we add it to the output string,and if it predicts the eos token we stop there.
we also store the decoder's attention outputs for display later.
"""

def evaluate(encoder,decoder,sentence,max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor=tensorFromSentence(input_lang,sentence)
        input_length=input_tensor.size()[0]
        encoder_hidden=encoder.initHidden()
        
        encoder_outputs=torch.zeros(max_length,encoder.hidden_size,device=device)
        
        for ei in range(input_length):
            encoder_output,encoder_hidden=encoder(input_tensor[ei],encoder_hidden)
            encoder_outputs[ei]+=encoder_output[0,0]
            
        decoder_input=torch.tensor([[sos_token]],device=device) # SOS
        decoder_hidden=encoder_hidden
        
        decoded_words=[]
        decoder_attentions=torch.zeros(max_length,max_length)
        
        for di in range(max_length):
            decoder_output,decoder_hidden,decoder_attention=decoder(decoder_input,decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == eos_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]
        
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
        
        
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
evaluateRandomly(encoder1, attn_decoder1) 

# visualiaing attention : plt.matshow(attentions)를 이용
output_words,attentions=evaluate(encoder1,attn_decoder1,"je suis parti")
plt.matshow(attentions.numpy())

def showAttention(input_sentence,output_words,attentions):
    # set up figure with colorbar
    fig=plt.figure()
    ax=fig.add_subplot(111)
    cax=ax.matshow(attentions.numpy(),cmap='bone')
    fig.colorbar(cax)
    
    # set up axes
    ax.set_xticklabels(['']+input_sentence.split(' ')+['<EOS>'],rotation=90)
    ax.set_yticklabels(['']+output_words)
    
    # show label at every tick 
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.show()
    
def evaluateAndShowAttention(input_sentence):
    output_words,attentions=evaluate(encoder1,attn_decoder1,input_sentence)
    print('input=',input_sentence)
    print('output=',' '.join(output_words))
    showAttention(input_sentence,output_words,attentions)

evaluateAndShowAttention("elle a cinq ans de moins que moi .")
evaluateAndShowAttention("elle est trop petit .")
evaluateAndShowAttention("je ne crains pas de mourir .")
evaluateAndShowAttention("c est un jeune directeur plein de talent .")

