import pandas as pd
import random
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup, RobertaTokenizer, RobertaModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import data_preprocess
from imblearn.over_sampling import SMOTE
import dataloader
import model
import train_eval
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(7)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device('cpu')
BERT_org = BertModel.from_pretrained("bert-base-uncased")

numerical_data = pd.read_csv('all_after_preprocessingStopwords.csv')


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
preprocessed_data, boolean_features, cate_features, numerical_features = data_preprocess.Data_preprocessing(numerical_data)

cate_num = len(cate_features)
bool_num =  len(boolean_features)
num_num = len(numerical_features)
print('The number of categorical feature is : {}, boolean feature is : {}, numerical feature is : {}'.format(cate_num, bool_num, num_num))

num_feature = cate_features + numerical_features + boolean_features
res_feature = ['description', 'price_range']
final_data = preprocessed_data[num_feature + res_feature]
X_train, X_test = train_test_split(final_data, test_size=0.1, random_state=13)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

strategy = {3:889, 4:797, 5:583}
sm = SMOTE(random_state=42,sampling_strategy=strategy)
X_res, y_res = sm.fit_resample(X_train[num_feature], X_train['price_range'])
X_res['price_range'] = y_res


def import_sentence(df, df_smote, path):
    label_3_sentences = []
    label_4_sentences = []
    label_5_sentences = []
    with open(path + 'label_3_sentence.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            label_3_sentences.append(line.strip())
    with open(path + 'label_4_sentence.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            label_4_sentences.append(line.strip())
    with open(path + 'label_5_sentence.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            label_5_sentences.append(line.strip())

    org_sentences = list(df['description'])
    new_sentences = []
    idx3, idx4, idx5 = 0, 0, 0

    for i in range(len(df), len(df_smote)):

        if df_smote.iloc[i]['price_range'] == 3:
            new_sentences.append(label_3_sentences[idx3])
            idx3 += 1
        elif df_smote.iloc[i]['price_range'] == 4:
            new_sentences.append(label_4_sentences[idx4])
            idx4 += 1
        elif df_smote.iloc[i]['price_range'] == 5:
            new_sentences.append(label_5_sentences[idx5])
            idx5 += 1

    df_smote['description'] = org_sentences + new_sentences
    return df_smote

df_smote = import_sentence(X_train, X_res, 'generated_sentence/')
Train_dataset = dataloader.Generate_dataset(df_smote, tokenizer, boolean_features,cate_features, numerical_features)
Test_dataset = dataloader.Generate_dataset(X_test, tokenizer, boolean_features,cate_features, numerical_features)

multi_mode = {'cate_num': cate_num, 'bool_num':bool_num, 'num_num':num_num}
Bert_regressor = model.BERT_Predictor(BERT_org, tokenizer,multi_mode = multi_mode).to(device)

num_epochs = 5
lr_rate = 5e-5
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
param_optimizer = list(Bert_regressor.named_parameters())
optimizer_grouped_parameters = [
        {
                'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay':0.01
        },
        {
                'params':[p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay':0.0
        }
]
total_steps = len(Train_dataloader) * num_epochs
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, optimizer_grouped_parameters[0]['params']), lr= lr_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps,num_training_steps=total_steps)

train_eval.train(Bert_regressor,optimizer, Train_dataloader,device,num_epochs)
fin_targets_range, fin_outputs_range = train_eval.evaluate_attacker(Bert_regressor, Test_dataloader)

