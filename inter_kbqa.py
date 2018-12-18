# -*- coding:utf-8 -*-

import os, sys, json
import re, math
import random
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import nltk

USE_CUDA = True

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 30
accept_hit = 20 # for validation step

# Class for make word vocabulary
class Lang :
    def __init__(self) :
        self.word2index = {'SOS' : 0, 'EOS' : 1, 'UNK' : 2, 'PAD' : 3}
        self.index2word = {0: 'SOS', 1 :'EOS', 2 : 'UNK', 3 :'PAD'}
        self.n_words = 4

    def word_index(self, sentence) :
        for word in sentence.split() :
            if word not in self.word2index :
                self.word2index[word] = len(self.word2index)

    def index_word(self) :
        self.n_words = len(self.word2index)   
        for word, index in self.word2index.items() :
            if index not in self.index2word :
                self.index2word[index] = word

    def word_to_index(self, word) :
        if word in self.word2index :
            return self.word2index[word]
        else :
            return self.word2index['UNK']

    def index_to_word(self, idx) :
        return self.index2word[idx]

# Encoder 
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        
    def forward(self, word_inputs, hidden):
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = torch.zeros(self.n_layers, 1, self.hidden_size)
        if USE_CUDA: hidden = hidden.cuda()
        return hidden

# Attention model
class Attn(nn.Module):
    def __init__(self, method, hidden_size, max_length=MAX_LENGTH):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        # Create variable to store attention weights
        attn_energies = torch.zeros(seq_len) # B x 1 x S
        if USE_CUDA: attn_energies = attn_energies.cuda()

        # Calculate attention weights for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0) #resize to 1 x 1 x seq_len
    
    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy, hidden = energy.view(-1), hidden.view(-1)
            energy = hidden.dot(energy)
            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy


# Interactive Attention Model
class InteractiveAttn(nn.Module):
    def __init__(self, method, hidden_size, max_length=MAX_LENGTH):
        super(InteractiveAttn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn_pool_q = nn.Linear(self.hidden_size, hidden_size)
            self.attn_q_hidden = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs, pool_output):
        seq_len = len(encoder_outputs)

        # Create variable to store attention weights
        attn_energies = torch.zeros(seq_len) # B x 1 x S
        if USE_CUDA: attn_energies = attn_energies.cuda()

        # Calculate attention weights for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i], pool_output)

        # Normalize energies to weights in range 0 to 1
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0) #resize to 1 x 1 x seq_len
    
    def score(self, hidden, encoder_output, pool_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            # First. Get energy between pooling output and encoder output
            energy = self.attn_pool_q(pool_output)
            energy, encoder_output = energy.view(-1), encoder_output.view(-1)
            #print('[1st energy size] ', energy.size(), encoder_output.size())
            attn_encoder = encoder_output.mul(energy)
            #print('[attn_encoder size] ',attn_encoder)

            # Second. Get energy between attn encoder output and hidden
            energy = self.attn_q_hidden(attn_encoder.view(-1))
            energy, hidden = energy.view(-1), hidden.view(-1)
            energy = hidden.dot(energy)

            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy

# Decoder for attention
class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.merge = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size * 2, output_size)
        
        if attn_model != 'none':
            self.attn_q = InteractiveAttn(attn_model, hidden_size)
            self.attn_kb = InteractiveAttn(attn_model, hidden_size)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs, kb_outputs):     
        # Get the embedding of the input word(last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        
        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2) #si
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        ### Attention Phase ###
        # Avg pooling for interactive attention
        avgPoolLayer = nn.AdaptiveAvgPool2d((1, hidden_size))
        pool_q = avgPoolLayer(encoder_outputs.transpose(0, 1))
        pool_kb = avgPoolLayer(kb_outputs.transpose(0, 1))

        # First attention from kb(pooling information) to question -> si * W_a * h_q * W_b * pool_kb
        # Calculate attention from current RNN state and all question outputs; apply to encoder outputs
        attn_weights_q = self.attn_q(rnn_output.squeeze(0), encoder_outputs, pool_kb)
        context_q = attn_weights_q.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N

        # Second attention from question(pooling information) to kb -> si * W_a * h_kb * W_b * pool_q
        # Calculate attention from current RNN state and all kb outputs; apply to kb outputs
        attn_weights_kb = self.attn_kb(rnn_output.squeeze(0), kb_outputs, pool_q)
        context_kb = attn_weights_kb.bmm(kb_outputs.transpose(0, 1)) # B x 1 x N

        ### Final Phase ###
        # Final output layer (next word prediction) using the RNN hidden state and context vectors
        rnn_output = rnn_output.squeeze(0) # (S=1) x B x N -> B x N
        context_q, context_kb = context_q.squeeze(1), context_kb.squeeze(1) # B x (S=1) x N -> B x N, B x (S=1) x N -> B x N
        context = self.merge(torch.cat((context_q, context_kb), 1))
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        
        # Return final output, hidden state, and attention weights
        return output, context, hidden, attn_weights_q, attn_weights_kb


def train(input_variable, kb_variable, target_variable, encoder, decoder, 
    encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio = 0.5, max_length=MAX_LENGTH):

    # Zero gradients of optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Get size of input question, kb and target sentences
    input_length = input_variable.size()[0]
    kb_length = kb_variable.size()[0]
    trg_length = target_variable.size()[0]

    # Run words through question encoder
    encoder_hidden = encoder.init_hidden().cuda()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Run word through kb encoder
    kb_hidden = encoder.init_hidden().cuda()
    kb_outputs, kb_hidden = encoder(kb_variable, kb_hidden)

    # Prepare input and output variables
    decoder_input = torch.LongTensor([[SOS_token]]).cuda()
    decoder_context = torch.zeros(1, decoder.hidden_size).cuda()
    decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder

    loss = 0 
    # If teacher forcing ratio is true(>0), train with next target input
    #user_teaching_ratio = random.random()

    if teacher_forcing_ratio != 0:# Teacher forcing       
        for di in range(trg_length):
            decoder_output, decoder_context, decoder_hidden, _, _ = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs, kb_outputs)
            loss += criterion(decoder_output, target_variable[di].view(1))
            decoder_input = target_variable[di] # Next target is next input

    else: # Without teacher forcing
        for di in range(1, trg_length):
            decoder_output, decoder_context, decoder_hidden, _, _ = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di].view(1))
            
            # Get word which has highest probability from output
            topv, topi = decoder_output.data.topk(1) # highest one
            ni = topi[0][0]
            
            decoder_input = Variable(torch.LongTensor([[ni]])) # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
            if ni == EOS_token: break

    # Backpropagation
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item() / trg_length

# Test
def test(input_variable, kb_variable, target_variable, encoder, decoder, max_length = MAX_LENGTH) :

    encoder_hidden = encoder.init_hidden()
    src_length = input_variable.size()[0]
    trg_length = target_variable.size()[0]

    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Run word through kb encoder
    kb_hidden = encoder.init_hidden().cuda()
    kb_outputs, kb_hidden = encoder(kb_variable, kb_hidden)

    decoder_attns_q, decoder_attns_kb = [], []
    decoder_hidden = encoder_hidden
    decoder_context = torch.zeros(1, decoder.hidden_size).cuda()
    decoder_outputs = torch.zeros(1, max_length).cuda()

    decoder_input = torch.LongTensor([[SOS_token]]).cuda()

    for i in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attn_q, decoder_attn_kb = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs, kb_outputs)
        decoder_attns_q.append(decoder_attn_q.data[0].tolist())
        decoder_attns_kb.append(decoder_attn_kb.data[0].tolist())
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        
        decoder_input = torch.LongTensor([[ni]]).cuda()

        if ni == EOS_token or i == max_length -1 : break

        decoder_outputs[0][i+1] = ni

    # Return generated decoder outputs, attentions
    return decoder_outputs, decoder_attns_q, decoder_attns_kb

def make_dictionary(path) :
    with open(path + 'source.txt', 'r', encoding = 'utf-8') as f :
        for nl_sent in f :
            input_lang.word_index(nl_sent.rstrip())

    with open(path + 'predicate.txt', 'r', encoding = 'utf-8') as f :
        for nl_sent in f :
            input_lang.word_index(nl_sent.rstrip())
    input_lang.index_word()

    with open(path + 'target.txt', 'r', encoding = 'utf-8') as f :
        nl_sents = f
        for nl_sent in f :
            target_lang.word_index(nl_sent.rstrip())

    target_lang.index_word()

def open_data(path) :
    src_pair, kb_pair, trg_pair = [], [], []

    with open(path + 'source.txt', 'r', encoding = 'utf-8') as f :
        for nl_sent in f :
            sent_list = []
            nl_sent = nl_sent.rstrip()
            for nl_word in nl_sent.split() :
                sent_list.append(input_lang.word_to_index(nl_word))
            src_pair.append(sent_list)

    with open(path + 'predicate.txt', 'r', encoding = 'utf-8') as f :
        for nl_sent in f :
            sent_list = []
            nl_sent = nl_sent.rstrip()
            for nl_word in nl_sent.split() :
                sent_list.append(input_lang.word_to_index(nl_word))
            kb_pair.append(sent_list)

    if 'test' not in path : 
        with open(path + 'target.txt', 'r', encoding = 'utf-8') as f :
            for nl_sent in f :
                sent_list = []
                nl_sent = nl_sent.rstrip()
                for nl_word in nl_sent.split() :
                    sent_list.append(target_lang.word_to_index(nl_word))
                sent_list.append(EOS_token)
                trg_pair.append(sent_list)
    else : 
        with open(path + 'target.txt', 'r', encoding = 'utf-8') as f :
            for nl_sent in f :
                sent_list = []
                nl_sent = nl_sent.rstrip().split()
                trg_pair.append(nl_sent)

        print('[Dataload] size of src_pair : %d, kb pair : %d, target pair : %d'%(len(src_pair), len(kb_pair), len(trg_pair)))
    return src_pair, kb_pair, trg_pair


################ Main ################

# Load dataset and make word embedding
print('[Dataload] Make word dicitionary ')

input_lang = Lang()
target_lang = Lang()

make_dictionary('dataset/train/')

with open('result/word2index.txt', 'w', encoding = 'utf8') as f:
    json.dump(target_lang.word2index, f)

with open('result/index2word.txt', 'w', encoding = 'utf8') as f:
    json.dump(target_lang.index2word, f)

print('[Dataload] Data Train / Valid / Test Data ')
train_src, train_kb, train_trg = open_data('dataset/train/')
valid_src, valid_kb, valid_trg = open_data('dataset/valid/')
test_src, test_kb, test_trg = open_data('dataset/test/')


attn_model = 'general'
hidden_size = 200
n_layers = 2
dropout_p = 0.0

# Initialize models
# question & kb triple share encoder
encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers)
decoder = AttnDecoderRNN(attn_model, hidden_size, target_lang.n_words, n_layers, dropout_p=dropout_p)

# Move models to GPU
#if USE_CUDA:
encoder.cuda()
decoder.cuda()

# Initialize optimizers and criterion
learning_rate = 0.0001
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

criterion = nn.NLLLoss()

n_epochs = 100

num_train = len(train_src)
num_valid = len(valid_src)
num_test = len(test_src)

print('input word size : %d, target word size : %d'%(input_lang.n_words, target_lang.n_words))
print('num of train : %d, num of valid : %d, num of test : %d'%(num_train, num_valid, num_test))
print('Start!')

hit_count = 0
min_loss = 999999

teacher_forcing_ratio = 0.5

for epoch in range(1, n_epochs + 1):
    # Enter Train Model
    encoder.train()
    decoder.train()

    train_loss = 0

    for i in range(num_train) :
        # Get training data for this cycle
        input_variable = torch.LongTensor(train_src[i]).cuda()
        kb_variable = torch.LongTensor(train_kb[i]).cuda()
        target_variable = torch.LongTensor(train_trg[i]).cuda()

        loss = train(input_variable, kb_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio)
        train_loss += loss

    print('Epoch : %d | train loss : %.2f'%(epoch, train_loss))

    if min_loss > train_loss :
        min_loss = train_loss
    elif min_loss < train_loss :
        hit_count += 1

    if hit_count == accept_hit :
        sys.exit(1)

    # Enter Eval mode
    encoder.eval()
    decoder.eval()

    # Test
    decoder_predicts, inter_attn_qs, inter_attn_kbs = [], [], []
    for i in range(num_test) :
        input_variable = torch.LongTensor(test_src[i]).cuda()
        kb_variable = torch.LongTensor(test_kb[i]).cuda()
        target_variable = torch.LongTensor(test_trg[i]).cuda()

        decoder_predict, attn_q, attn_kb = test(input_variable, kb_variable, target_variable, encoder, decoder)
        decoder_predicts.append(decoder_predict.data.tolist())
        inter_attn_qs.append(attn_q)
        inter_attn_kbs.append(attn_kb)

    # Save entire results & model
    bleu_scores = 0
    with open('result/test_predicts.json', 'w', encoding = 'utf-8') as f:
        #json.dump(decoder_predicts, f)
        for _idx in range(len(decoder_predicts)) :
            pred_sent = ''
            trg_word = test_trg[_idx]
            for word in decoder_predicts[_idx][0] :
                if word == 0 or word == 1:
                    continue
                word = input_lang.index2word[int(word)]
                pred_sent = pred_sent + ' ' + word
                pred_sent = re.sub(' +', ' ', pred_sent)

                # If you want to measure bleu score
                sf = nltk.translate.bleu_score.SmoothingFunction()
                bleu_score =  nltk.translate.bleu_score.sentence_bleu([trg_word], pred_sent.split(), smoothing_function = sf.method1)
                bleu_scores += bleu_score

            f.write(sent + '\n')

    print('test bleu score : %.4f'%(bleu_score))
    
    # If you want to save attnetion weights ... 
    #with open('result/test_attn_q.json', 'w', encoding = 'utf-8') as f:
    #    json.dump(inter_attn_qs, f)
    #with open('result/test_attn_kb.json', 'w', encoding = 'utf-8') as f:
    #    json.dump(inter_attn_kbs, f)

    torch.save(encoder, 'model/encoder_q.pt')
    torch.save(decoder, 'model/decoder_kb.pt')
