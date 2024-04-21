import pickle
import numpy as np
import pandas as pd
import torch
import json

from torch import nn
from transformers import BertModel, BertTokenizer

def get_keyword_vector(keywords):
    keyword_vector = [0.]*len(keyword_2_id)
    for keyword in keywords:
        id_ = keyword_2_id.get(keyword)
        if id_ is not None:
            keyword_vector[id_] = 1.
    return keyword_vector

keyword_2_id, id_2_keyword = [], []
with open('keywords.pickle', 'rb') as handle:
    keyword_2_id, id_2_keyword = pickle.load(handle)

with open("rubert_cased_L-12_H-768_A-12_pt/bert_config.json", "r") as read_file, open("rubert_cased_L-12_H-768_A-12_pt/config.json", "w") as conf:
    file = json.load(read_file)
    conf.write(json.dumps(file))

tokenizer = BertTokenizer.from_pretrained('rubert_cased_L-12_H-768_A-12_pt')
model = BertModel.from_pretrained('rubert_cased_L-12_H-768_A-12_pt', output_hidden_states = True)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output['last_hidden_state']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 64
learning_rate = 1e-3
num_epochs = 10
num_workers = 2 if device.type == "cuda" else None
number_keywords = len(keyword_2_id)
embedding_dim = 768

class KeywordClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(KeywordClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, embeddings):
        x = self.fc1(embeddings)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        logits = self.fc3(x)
        return self.sigmoid(logits)
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()
classifier_model = KeywordClassifier(embedding_dim, number_keywords).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(classifier_model.parameters(), lr=1e-3)

checkpoint = torch.load(f"checkpoint_a.pt", map_location=torch.device('cpu'))
classifier_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
loss_history = checkpoint['loss_history']


def get_predicted_keywords(logits, id_2_keyword, top_keywords=10):
    top_k_values, top_k_indices = logits.topk(top_keywords, dim=1)
    predicted_keywords = []
    for i in range(logits.shape[0]):
        predicted_keywords.append([id_2_keyword[idx.item()] for idx in top_k_indices[i]])
    return predicted_keywords

def predict_keywords(title, description, text, top_keywords=10):
    model.eval()
    classifier_model.eval()
    with torch.no_grad():
        text = "; ".join([title, description, ""])
        tokenized = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=100)
        items = {k: v.reshape(1, -1).to(device) for k, v in tokenized.items()}
        outputs = model(**items)
        embeddings = mean_pooling(outputs, items['attention_mask'])
        # print("embeddings: ", embeddings[0][:10])
        logits = classifier_model(embeddings).cpu() # expit
    # print("Logits", logits[0][:10])
    return get_predicted_keywords(logits, id_2_keyword, top_keywords)[0]

# EXAMPLE:
# predict_keywords(title, description, "", top_keywords=5)
