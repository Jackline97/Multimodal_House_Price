import torch

class BERT_Predictor(torch.nn.Module):
    def __init__(self, pretrained_model, tokenizer, predictor_size=1, multi_mode=None):
        super(BERT_Predictor, self).__init__()

        self.bert = pretrained_model
        self.PAD_INDEX = tokenizer.pad_token_id

        self.final_embed = 36

        if multi_mode is not None:
            self.cat_num = multi_mode['cate_num']
            self.bool_num = multi_mode['bool_num']
            self.num_num = multi_mode['num_num']

        self.cat_encoder = torch.nn.Linear(self.cat_num, self.final_embed)
        self.bool_encoder = torch.nn.Linear(self.bool_num, self.final_embed)
        self.num_encoder = torch.nn.Linear(self.num_num, self.final_embed)
        self.des_encoder = torch.nn.Linear(768, self.final_embed)

        self.multihead_attn = torch.nn.MultiheadAttention(self.final_embed, 12, dropout=0.2, batch_first=True)
        self.layer_norm = torch.nn.LayerNorm(self.final_embed)
        self.dropout = torch.nn.Dropout(0.2)

        self.predictor = torch.nn.ModuleList()
        for _ in range(predictor_size):
            self.predictor.append(torch.nn.Linear(self.final_embed, self.final_embed))
            self.predictor.append(torch.nn.ReLU())
            self.predictor.append(torch.nn.Dropout(0.2))

        self.prediction = torch.nn.Linear(self.final_embed, 6)

    def forward(self, input_ids, cat_feature=None, numerical_feature=None, bool_feature=None):
        attention_mask = (input_ids != self.PAD_INDEX).type(torch.uint8)

        outputs = self.bert(input_ids, attention_mask=attention_mask)
        final_output = outputs[0]

        cat_feature = cat_feature.float()
        numerical_feature = numerical_feature.float()
        bool_feature = bool_feature.float()

        cat_feature_final = self.cat_encoder(cat_feature).unsqueeze(1)
        numerical_feature_final = self.num_encoder(numerical_feature).unsqueeze(1)
        bool_feature_final = self.bool_encoder(bool_feature).unsqueeze(1)
        des_output = self.des_encoder(final_output)

        numerical = torch.cat((cat_feature_final, numerical_feature_final, bool_feature_final, des_output), 1)

        attn_output, _ = self.multihead_attn(numerical, numerical, numerical)
        attn_output = self.dropout(attn_output)
        attn_output = attn_output
        attn_output = self.layer_norm(attn_output + numerical)

        pooled_output = torch.mean(attn_output, dim=1)

        for layer in self.predictor:
            pooled_output = layer(pooled_output)

        prediction = self.prediction(pooled_output)
        return prediction