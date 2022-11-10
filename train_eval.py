from tqdm import tqdm
import torch

def train(model,optimizer, train_iter,device,num_epochs):
    model.train()
    R2_final = []
    MSE = []
    R2_val = 0
    MSE_val = 0
    step = 0
    for _ in range(num_epochs):
        for data in tqdm(train_iter):
            optimizer.zero_grad()
            source = data['description'].to(device)
            target_range = data['label_range'].to(device).unsqueeze(-1)
            cat_feature = data['cate_feature'].to(device)
            numerical_feature = data[ 'numerical_feature'].to(device)
            bool_feature = data['boolean_feature'].to(device)
            cls = model(input_ids=source,cat_feature = cat_feature, numerical_feature = numerical_feature, bool_feature = bool_feature)
            loss = torch.nn.CrossEntropyLoss()(cls, target_range.long().squeeze())
            loss.backward()
            optimizer.step()
            scheduler.step()
        print('CLS loss is:{}'.format(float(loss)))

def evaluate_attacker(model, test_iter):
    fin_targets_range = []
    fin_outputs_range = []
    with torch.no_grad():
        model.eval()
        for data in tqdm(test_iter):
            source = data['description'].to(device)
            target_range = data['label_range'].to(device).unsqueeze(-1)
            cat_feature = data['cate_feature'].to(device)
            numerical_feature = data[ 'numerical_feature'].to(device)
            bool_feature = data['boolean_feature'].to(device)
            cls = model(input_ids=source,cat_feature = cat_feature, numerical_feature = numerical_feature, bool_feature = bool_feature)
            try:
                fin_targets_range.extend(target_range.tolist())
                fin_outputs_range.extend(torch.argmax(cls,1).tolist())
            except:
                print('---')
    return fin_targets_range, fin_outputs_range