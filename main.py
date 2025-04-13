

import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.optim as optim
import nltk
from indicnlp.tokenize import indic_tokenize

filePath="/content/drive/MyDrive/corpus/Dataset_English_Hindi.csv"

corpus=pd.read_csv(filePath)

# pd.options.display.max_colwidth=100
pd.set_option('display.max_colwidth', None)

"""# Text Preprocesssing"""

corpus.sample(5)

corpus.info()

corpus.isnull().sum()

corpus.dropna(inplace=True)
corpus.drop_duplicates(inplace=True)
corpus=corpus.reset_index(drop=True)
corpus

for i, eng,hin in corpus.itertuples():
  if eng.isspace() or hin.isspace():
    corpus.drop(i, inplace=True)
    print(eng,"\n",hin,"\n\n")

def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'',text)

corpus['English'] = corpus['English'].apply(remove_url)
corpus['Hindi'] = corpus['Hindi'].apply(remove_url)

def remove_html_tags(text):
    pattern =r'<.*?>'
    text = re.sub(pattern,'',text)
    return text
corpus['English']=corpus['English'].apply(remove_html_tags)
# remove_html_tags("hello are , your there?")

def preprocess_text(text, language='english'):
    if not isinstance(text, str):
        return text

    if language == 'english':
        pattern = re.compile(r'[^a-zA-Z0-9\s]')
        return pattern.sub(r'', text)
    elif language == 'hindi':
        pattern = re.compile(r'[^\u0900-\u097F\s]')
        return pattern.sub(r'', text)
    else:
        raise ValueError("Unsupported Language, Supported languages are 'english' and 'hindi'")

corpus['English'] = corpus['English'].apply(lambda x: preprocess_text(x, language='english'))
corpus['Hindi'] = corpus['Hindi'].apply(lambda x: preprocess_text(x, language='hindi'))

# type(corpus.iloc[0]["English"])

"""## Tokenization"""



nltk.download('punkt_tab')

# nltk.word_tokenize(corpus.iloc[1456]["English"].lower())
# indic_tokenize.trivial_tokenize_indic(corpus.iloc[1456]["Hindi"])

PAD_TOKEN = 0
UNK_TOKEN = 1
SOS_TOKEN = 2
EOS_TOKEN = 3

eng_vocab={'<PAD>':PAD_TOKEN, '<UNK>':UNK_TOKEN,"<SOS>": SOS_TOKEN, "<EOS>": EOS_TOKEN,}
hin_vocab={'<PAD>':PAD_TOKEN ,'<UNK>':UNK_TOKEN,"<SOS>": SOS_TOKEN, "<EOS>": EOS_TOKEN,}

def vocabGen(row):
  eng_token=nltk.word_tokenize(row["English"].lower())
  hin_token=indic_tokenize.trivial_tokenize_indic(row["Hindi"])

  for token in eng_token:
    if token not in eng_vocab:
      eng_vocab[token]=len(eng_vocab)
  for token in hin_token:
    if token not in hin_vocab:
      hin_vocab[token]=len(hin_vocab)

corpus.apply(vocabGen,axis=1)

len(eng_vocab) ,len(hin_vocab)

def text2idx(sentence, lang,vocab):
  if lang=="eng":
    sentence_token=nltk.word_tokenize(sentence.lower())
  else:
    sentence_token=indic_tokenize.trivial_tokenize(sentence)
  indexed_text = []
  for token in sentence_token:
    if token in vocab:
      indexed_text.append(vocab[token])
    else:
      indexed_text.append(UNK_TOKEN)
  indexed_text.append(EOS_TOKEN)

  return indexed_text

text2idx(corpus.iloc[1234]["English"],"eng",eng_vocab)
# corpus.iloc[1234]

"""## Dataset preparation"""


class MTDataset(Dataset):

  def __init__(self, df, eng_vocab,hin_vocab):
    self.df = df
    self.eng_vocab = eng_vocab
    self.hin_vocab= hin_vocab

  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self, index):

    numerical_eng = text2idx(self.df.iloc[index]['English'],"eng", self.eng_vocab)
    numerical_hin = text2idx(self.df.iloc[index]['Hindi'], "hin", self.hin_vocab)
    encoder_inputs =torch.tensor(numerical_eng[:-1],dtype=torch.long)
    numerical_hin=torch.tensor(numerical_hin,dtype=torch.long)
    decoder_targets=numerical_hin

    return encoder_inputs, decoder_targets

def collate_fn(batch):
    encoder_inputs, decoder_targets = zip(*batch)

    encoder_inputs = pad_sequence(encoder_inputs, batch_first=True, padding_value=0)
    decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=0)

    return encoder_inputs, decoder_targets

dataset = MTDataset(corpus, eng_vocab, hin_vocab)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

BATCH_SIZE=16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# index2word
hin_idx2word={v:k for k,v in hin_vocab.items()}

maxLen={"l":0}
def maxLength(row):
  maxLen["l"]=max(max(maxLen["l"],len(row["English"].split())),max(maxLen["l"],len(row["Hindi"].split())))

corpus.apply(maxLength,axis=1)
combined_maxLen=maxLen["l"]

combined_maxLen

"""# MODEL"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

class EncoderRNN(nn.Module):
  def __init__(self,eng_vocab_size,hidden_size):
    super().__init__()
    self.embedding=nn.Embedding(eng_vocab_size,embedding_dim=256)
    self.lstm = nn.LSTM(256, hidden_size, batch_first=True, bidirectional=True)

  def forward(self, encoder_inputs):
    embedded_input = self.embedding(encoder_inputs)
    output, (hidden_state,cell_state) = self.lstm(embedded_input)
    # print(hidden_state.shape)
    hidden_state = torch.cat((hidden_state[0,:,:], hidden_state[1,:,:]), dim=1).unsqueeze(0)
    cell_state = torch.cat((cell_state[0,:,:], cell_state[1,:,:]), dim=1).unsqueeze(0)

    return output, hidden_state,cell_state

class Attention(nn.Module):
  def __init__(self,hidden_size):
    super().__init__()
    self.l = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)
    self.o = nn.Linear(hidden_size * 2, 1, bias=False)


  def forward(self,encoder_output,decoder_hidden):
    decoder_hidden=decoder_hidden.squeeze(0)
    # print(encoder_output.shape , decoder_hidden.shape)
    decoder_hidden=decoder_hidden.unsqueeze(1).repeat(1,encoder_output.size(1),1)
    # print(encoder_output.shape , decoder_hidden.shape)
    combined=torch.cat((encoder_output,decoder_hidden),dim=2)

    x = torch.tanh(self.l(combined))
    attention = self.o(x).squeeze(2)
    attention_weights = F.softmax(attention, dim=1)
    context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_output).squeeze(1)
    return context_vector, attention_weights

class AttentionDecoder(nn.Module):
  def __init__(self,hin_vocab_size,hidden_size):
    super().__init__()
    self.attention = Attention(hidden_size)
    self.embedding=nn.Embedding(hin_vocab_size,embedding_dim=256)
    self.lstm = nn.LSTM(256+hidden_size, hidden_size, batch_first=True)
    self.dropout = nn.Dropout(0.5)
    self.out = nn.Linear(hidden_size, hin_vocab_size)
    self.attention_matrix = []
  def forward(self,input, encoder_hidden,encoder_cell,encoder_output ,store_attention=False):

      output = self.dropout(self.embedding(input))
      context_vector,attention_weights=self.attention(encoder_output,encoder_hidden)
      if store_attention:
        self.attention_matrix.append(attention_weights.detach().cpu().numpy())
      # print(input.shape,context_vector.unsqueeze(1).shape)
      lstm_input = torch.cat((output, context_vector.unsqueeze(1)), dim=2)
      # print(lstm_input.shape)
      output, (hidden, cell) = self.lstm(lstm_input, (encoder_hidden, encoder_cell))
      output = self.out(output)
      return output, hidden, cell


"""# Translation"""

HIDDEN_SIZE=128
eng_vocab_size = len(eng_vocab)
hin_vocab_size = len(hin_vocab)

encoder = EncoderRNN(eng_vocab_size,HIDDEN_SIZE).to(device)
decoder = AttentionDecoder(hin_vocab_size,HIDDEN_SIZE*2).to(device)

def translate(encoder,decoder,sentence,eng_vocab,hin_idx2word,maxLength=30):
  encoder.eval()
  decoder.eval()
  with torch.inference_mode():
    tokenized_sent=text2idx(sentence,"eng",eng_vocab)[:-1]
    input_tensor= torch.tensor(tokenized_sent,dtype=torch.long)
    input_tensor = input_tensor.view(1, -1).to(device)
    encoder_output , encoder_hidden,encoder_cell=encoder(input_tensor)

    decoder_input=torch.tensor([[SOS_TOKEN]]).to(device)
    decoder_hidden=encoder_hidden
    decoder_cell=encoder_cell
    decoded_words=[]
    for di in range(maxLength):
      logits,decoder_hidden,decoder_cell=decoder(decoder_input,decoder_hidden,decoder_cell,encoder_output,store_attention=True)
      logits=logits.reshape(logits.size(0),-1)
      next_token=logits.argmax(axis=1)
      decoder_input=torch.tensor([[next_token]]).to(device)
      if next_token.item()==EOS_TOKEN:
        break
      else:
        decoded_words.append(hin_idx2word[next_token.item()])
    return ' '.join(decoded_words)

sentence="how are you"
translated_sent=translate(encoder,decoder,sentence,eng_vocab,hin_idx2word)
print("Hindi translation: ",translated_sent)


"""# Training"""

len(train_loader)

loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
encoder_optimizer=optim.AdamW(encoder.parameters())
decoder_optimizer=optim.AdamW(decoder.parameters())

epochs=4

encoder.train()
decoder.train()
for epoch in range(epochs):
  for i,(encoder_input, decoder_target) in enumerate(train_loader):
    input_tensor, target_tensor = encoder_input.to(device), decoder_target.to(device)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length = target_tensor.size(1)

    encoder_output, encoder_hidden ,encoder_cell= encoder(input_tensor)

    decoder_input = torch.full((BATCH_SIZE, 1),SOS_TOKEN, dtype=torch.long).to(device)
    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell

    batch_loss = 0
    for di in range(target_length):
      logits, decoder_hidden,decoder_cell  = decoder(decoder_input, decoder_hidden,decoder_cell,encoder_output)
      logits=logits.reshape(logits.size(0),-1)
      batch_loss += loss_fn(logits, target_tensor[:,di])
      decoder_input = target_tensor[:, di].reshape(BATCH_SIZE, 1)
    batch_loss /= target_length
    batch_loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    if i % 100 == 0:
        print(f'Epoch {epoch}, Batch {i}, Loss: {batch_loss.item() / target_length:.4f}')


print("done")

