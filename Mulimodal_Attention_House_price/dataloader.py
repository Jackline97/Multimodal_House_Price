import torch


class Generate_dataset(Dataset):
    def __init__(self, data_df, tokenizer, boolean_features, cate_features, numerical_features):
        self.MAX_SEQ_LEN = 128
        self.data = data_df
        self.tokenizer = tokenizer
        self.boolean = boolean_features
        self.cate = cate_features
        self.numerical = numerical_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        boolean_feature = torch.tensor(list(self.data.iloc[idx][self.boolean]))
        cate_feature = torch.tensor(list(self.data.iloc[idx][self.cate]))
        numerical_feature = torch.tensor(list(self.data.iloc[idx][self.numerical]))
        description = self.data.iloc[idx]['description']
        word_encode = torch.tensor(
            self.tokenizer.encode(text=description, max_length=self.MAX_SEQ_LEN, padding='max_length', truncation=True))
        #         label = torch.tensor(float(self.data.iloc[idx]['price']))
        label_range = torch.tensor(int(self.data.iloc[idx]['price_range']))

        return {'description': word_encode, 'numerical_feature': numerical_feature, 'cate_feature': cate_feature,
                'boolean_feature': boolean_feature, 'label_range': label_range}


