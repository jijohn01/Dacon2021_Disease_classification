import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import StratifiedKFold
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss

import warnings

warnings.filterwarnings("ignore")

class CustomDataset(Dataset):
    def __init__(self, files, labels=None, mode='train', transform=None):
        self.mode = mode
        self.files = files
        if mode == 'train':
            self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        if self.mode == 'train':
            img = Image.open('train_imgs/' + self.files[i])

            if self.transform:
                img = self.transform(img)

            return {
                'img': torch.tensor(img, dtype=torch.float32).clone().detach(),
                'label': torch.tensor(self.labels[i], dtype=torch.long)
            }
        else:
            img = Image.open('test_imgs/' + self.files[i])
            if self.transform:
                img = self.transform(img)

            return {
                'img': torch.tensor(img, dtype=torch.float32).clone().detach(),
            }

mytransform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(244),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=(0, 360)),
    transforms.RandomPerspective(),
    transforms.ToTensor(),
])

myvaltransform =transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

class CNN_Model(nn.Module):
    def __init__(self, class_n, rate=0.2):
        super(CNN_Model, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b7')#현 state of art
        self.dropout = nn.Dropout(rate)
        self.output_layer = nn.Linear(in_features=1000, out_features=class_n, bias=True)

    def forward(self, inputs):
        output = self.output_layer(self.dropout(self.model(inputs)))
        return output


def train_step(model, batch_item, epoch, batch, training,class_weight=None):
    img = batch_item['img'].to(device)
    label = batch_item['label'].to(device)
    if training is True:
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(img)
            loss = criterion(output, label,class_weight=class_weight)
        loss.backward()
        optimizer.step()

        return loss
    else:
        model.eval()
        with torch.no_grad():
            output = model(img)
            loss = criterion(output, label,class_weight=class_weight)

        return loss

def predict(models,dataset):
    for fold,model in enumerate(models):
        model.eval()
    tqdm_dataset = tqdm(enumerate(dataset))
    training = False
    results = []
    for batch, batch_item in tqdm_dataset:
        img = batch_item['img'].to(device)
        for fold,model in enumerate(models):
            with torch.no_grad():
                if fold ==0:
                    output = model(img)
                else:
                    output = output+model(img)
        output = 0.2*output
        output = torch.tensor(torch.argmax(output, axis=-1), dtype=torch.int32).cpu().numpy()
        results.extend(output)
    return results

if __name__ == '__main__':
    train_total = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    device = torch.device("cuda")
    batch_size = 8
    class_n = len(train_total['disease_code'].unique())
    learning_rate = 2e-4
    epochs = 300
    folds = 5
    random_seed=100
    save_path = 'models/model_fin_try.'
    kfold = StratifiedKFold(n_splits=folds,shuffle=True,random_state=random_seed)
    class_weight=torch.FloatTensor(1/train_total['disease_code'].value_counts()).cuda()

    torch.manual_seed(random_seed)

    train_dataset = CustomDataset(train_total['img_path'].str.split('/').str[-1].values, train_total['disease_code'].values,
                                  transform=mytransform)
    train_dataset = CutMix(train_dataset,num_class=7,beta=1.0,prob=0.5,num_mix=2)
    valid_dataset = CustomDataset(train_total['img_path'].str.split('/').str[-1].values, train_total['disease_code'].values,
                                  transform=myvaltransform)
    test_dataset = CustomDataset(test['img_path'].str.split('/').str[-1], labels=None, mode='test',
                                 transform=myvaltransform)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=3, shuffle=False)

    k_loss_plot, k_val_loss_plot = [], []

    for fold,(train_idx,valid_idx) in enumerate(kfold.split(train_dataset,train_total["disease_code"])):
        train_subsampler = SubsetRandomSampler(train_idx)
        valid_subsampler = SubsetRandomSampler(valid_idx)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=3, sampler=train_subsampler)
        val_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=3, sampler=valid_subsampler)

        model = CNN_Model(class_n).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = CutMixCrossEntropyLoss(True)

        loss_plot, val_loss_plot = [], []

        for epoch in range(epochs):
            total_loss, total_val_loss = 0, 0
            tqdm_dataset = tqdm(enumerate(train_dataloader))
            training = True
            for batch, batch_item in tqdm_dataset:
                batch_loss = train_step(model,batch_item, epoch, batch, training,class_weight=class_weight)
                total_loss += batch_loss

                tqdm_dataset.set_postfix({
                    'Epoch': epoch + 1,
                    'Loss': '{:06f}'.format(batch_loss.item()),
                    'Total Loss': '{:06f}'.format(total_loss / (batch + 1))
                })
            loss_plot.append((total_loss / (batch + 1)).cpu().item())

            tqdm_dataset = tqdm(enumerate(val_dataloader))
            training = False
            for batch, batch_item in tqdm_dataset:
                batch_loss = train_step(model,batch_item, epoch, batch, training)
                total_val_loss += batch_loss

                tqdm_dataset.set_postfix({
                    'Epoch': epoch + 1,
                    'Val Loss': '{:06f}'.format(batch_loss.item()),
                    'Total Val Loss': '{:06f}'.format(total_val_loss / (batch + 1))
                })
            val_loss_plot.append((total_val_loss / (batch + 1)).cpu().item())

            if np.min(val_loss_plot) == val_loss_plot[-1]:
                torch.save(model.state_dict(), save_path+str(fold)+".pt")

        k_loss_plot.append(min(loss_plot))
        k_val_loss_plot.append(min(val_loss_plot))

    print("Train Loss: ",np.mean(k_loss_plot),", Valid Loss: ",np.mean(k_val_loss_plot))

    models = []
    for i in range(5):
        model = CNN_Model(class_n).to(device)
        model.load_state_dict(torch.load(save_path+str(i)+".pt"))
        models.append(model)
    preds = predict(models,test_dataloader)

    submission = pd.read_csv('sample_submission.csv')
    submission.iloc[:, 1] = preds
    submission.to_csv('submission.csv', index=False)