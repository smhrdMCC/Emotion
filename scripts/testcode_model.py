import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np

# KoBERT
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
# DisstilBERT
from transformers import DistilBertModel

# GPU CUDA
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

# Load model
checkpoint=torch.load('../assets/saved_model.pt')

# Recognize metadata
max_len = checkpoint['max_len']
batch_size = checkpoint['batch_size']
model_name = checkpoint['model_name']

# Load KoBERT model
bertmodel, vocab = get_pytorch_kobert_model()

# If this is DistilBERT model:
if checkpoint['model_name']=="distilbert":
    distilbert_model = DistilBertModel.from_pretrained('monologg/distilkobert')
    bertmodel = distilbert_model

# BERT class
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))
    
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=7,   # Modify the number of the class
                 dr_rate=None,  
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.max_len = max_len
        self.batch_size = batch_size
        self.model_name = model_name
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        if checkpoint['model_name']=="bert":
            pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))[1]
        elif checkpoint['model_name']=="distilbert":
            pooler = self.bert(input_ids = token_ids, attention_mask = attention_mask.float().to(token_ids.device))[0][:,0]
            # Distilbert does not return the pooler, so extract pooler from last_hidden_state
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

# Define model    
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

# Get tokenizer
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

def predict(predict_sentence):

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=0)
    
    model.eval()
    with torch.no_grad():

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)

            valid_length= valid_length
            label = label.long().to(device)

            out = model(token_ids, valid_length, segment_ids)
            
            for i in out:
                logits=i
                logits = logits.detach().cpu().numpy()

                second_max_index = np.argpartition(logits, -2)[-2]
                second_max_value = logits[second_max_index]

                if np.argmax(logits) == 0:
                    prediction = "불안"
                elif np.argmax(logits) == 1:
                    prediction = "당황"
                elif np.argmax(logits) == 2:
                    prediction = "분노"
                elif np.argmax(logits) == 3:
                    prediction = "슬픔"
                elif np.argmax(logits) == 4:
                    prediction = "중립"
                elif np.argmax(logits) == 5 and (logits[5]/2) < second_max_value:
                    prediction = "행복"
                elif np.argmax(logits) == 5:
                    prediction = "더 행복"
                elif np.argmax(logits) == 6:
                    prediction = "혐오"

                print(logits)

# Load model
model.load_state_dict(checkpoint['model_state_dict'])

# Put The Data In!
while True:
    sentence = input("input is : ")
    if sentence == "" :
        break
    predict(sentence)
    print("\n")