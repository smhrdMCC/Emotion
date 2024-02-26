import requests
import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np

# KoBERT
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
# DistilBERT
from transformers import DistilBertModel

# Flask
from flask import Flask, request, jsonify

# # MariaDB
# import mariadb
# import sys

app = Flask(__name__)

# Random(?) seeds
seed_number = 52
torch.manual_seed(seed_number)
np.random.seed(seed_number)

# GPU CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
checkpoint = torch.load('../assets/saved_model.pt')

# Recognize metadata
max_len = checkpoint['max_len']
batch_size = checkpoint['batch_size']
model_name = checkpoint['model_name']

# Load KoBERT model
bertmodel, vocab = get_pytorch_kobert_model()

# If this is DistilBERT model:
if checkpoint['model_name'] == "distilbert":
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
                 hidden_size=768,
                 num_classes=7,  # Modify the number of the class
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.max_len = max_len
        self.batch_size = batch_size
        self.model_name = model_name

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        if checkpoint['model_name'] == "bert":
            pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(), attention_mask=attention_mask.float().to(token_ids.device))[1]
        elif checkpoint['model_name'] == "distilbert":
            pooler = self.bert(input_ids=token_ids, attention_mask=attention_mask.float().to(token_ids.device))[0][:, 0]
            # Distilbert does not return the pooler, so extract pooler from last_hidden_state
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

# Define model
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

# Get tokenizer
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
logits=0
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

            valid_length = valid_length
            label = label.long().to(device)

            out = model(token_ids, valid_length, segment_ids)

            emotion_priority=[]
            # logits = [불안, 당황, 분노, 슬픔, 중립, 행복, 혐오]
            emotion_labels = {0: "불안",
                              1: "당황",
                              2: "분노",
                              3: "슬픔",
                              4: "중립",
                              5: "기쁨",
                              6: "혐오",
                              7: "행복"}
            for logits in out:
                logits = logits.detach().cpu().numpy()

                predicted_emotion_index = np.argmax(logits)
                emotion_priority.append(emotion_labels[predicted_emotion_index])

                second_max_index = np.argpartition(logits, -2)[-2]
                emotion_priority.append(emotion_labels[second_max_index])

                if ((logits[predicted_emotion_index]*2) > logits[second_max_index]) & (predicted_emotion_index==5):
                    emotion_priority[0] = "행복"
                # print (logits)
        return emotion_priority[0]

# Send data
@app.route('/sendBert', methods=['POST'])

# If json data has received make json from data
def predict_emotion():
    recieved_data = request.get_json()
    sentence = recieved_data
    predicted_result = predict(sentence)
    print(predicted_result)
    return jsonify({'diaryContent':predicted_result})

    # If you want to whole infomation
    # return jsonify({'created_at': recieved_data['created_at'],
    #                 'emotion_classification': predicted_result,
    #                 'user_email': recieved_data['user_email']})


# # If you have to send json data
# def send_data(predicted_result):
#     data = {'predicted_result': predicted_result}
#     response = requests.post('http://10.0.2.2:8100/', json=data)
#     if response.status_code == 200:
#         print('Data sent successfully')
#     else:
#         print('Failed to send data:', response.status_code)


# # mariadb code - if you have to...
# # Save data to MariaDB
# def save_to_database(predicted_result):
#     try:
#         # Connect to MariaDB
#         conn = mariadb.connect(
#             user="mcc1234",
#             password="1234mcc",
#             host="project-db-campus.smhrd.com",
#             port=3308,
#             database="tb_feedback"
#         )
#         cur = conn.cursor()
#         cur.execute("INSERT INTO emotion_classification (result) VALUES (?)", (predicted_result,))
#         conn.commit()
#         print('Data saved to MariaDB successfully')
        
#     except mariadb.Error as e:
#         print(f"Error connecting to MariaDB: {e}")
#         sys.exit(1)
        
#     finally:
#         # Close the database connection
#         if conn:
#             conn.close()

# # Modify the send_data function to also save the predicted result to MariaDB
# def send_data(predicted_result):
#     # Send the predicted result back to the Flask app
#     data = {'emotion_classification': predicted_result}
#     response = requests.post('http://10.0.2.2:8100/', json=data)
#     if response.status_code == 200:
#         print('Data sent successfully to app')
#     else:
#         print('Failed to send data to app:', response.status_code)
    
#     # Save the predicted result to MariaDB
#     save_to_database(predicted_result)

# Define HOST
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8120)
