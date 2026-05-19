"""Sentiment Analysis — IMDB Reviews (PyTorch Bidirectional LSTM)"""
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets as tv_datasets
import numpy as np, matplotlib.pyplot as plt, seaborn as sns, re
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from collections import Counter
import warnings; warnings.filterwarnings("ignore")

VOCAB_SIZE,MAX_LEN,EMBED_DIM,HIDDEN_DIM=20000,200,64,128
BATCH_SIZE,EPOCHS,LR=64,8,0.001
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

train_data=tv_datasets.IMDB(root="./data",split="train",download=True)
test_data =tv_datasets.IMDB(root="./data",split="test", download=True)

def load_imdb(ds):
    texts,labels=[],[]
    for label,text in ds:
        texts.append(text); labels.append(1 if label==2 else 0)
    return texts,labels

train_texts,train_labels=load_imdb(train_data)
test_texts,test_labels=load_imdb(test_data)
print(f"Train: {len(train_texts):,}  Test: {len(test_texts):,}")

def clean(t): return re.sub(r"[^a-z\s]"," ",re.sub(r"<.*?>"," ",t.lower())).split()
all_tok=[t for txt in train_texts for t in clean(txt)]
vocab=["<PAD>","<UNK>"]+[w for w,_ in Counter(all_tok).most_common(VOCAB_SIZE-2)]
w2i={w:i for i,w in enumerate(vocab)}

def encode(text,ml=MAX_LEN):
    ids=[w2i.get(t,1) for t in clean(text)[:ml]]; ids+=[0]*(ml-len(ids)); return ids

class IMDBDataset(Dataset):
    def __init__(self,texts,labels):
        self.X=torch.tensor([encode(t) for t in texts],dtype=torch.long)
        self.y=torch.tensor(labels,dtype=torch.float)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.X[i],self.y[i]

train_loader=DataLoader(IMDBDataset(train_texts,train_labels),batch_size=BATCH_SIZE,shuffle=True)
test_loader =DataLoader(IMDBDataset(test_texts, test_labels), batch_size=BATCH_SIZE,shuffle=False)

class SentimentLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(VOCAB_SIZE,EMBED_DIM,padding_idx=0)
        self.lstm=nn.LSTM(EMBED_DIM,HIDDEN_DIM,batch_first=True,bidirectional=True,num_layers=2,dropout=0.3)
        self.fc=nn.Sequential(nn.Linear(HIDDEN_DIM*2,64),nn.ReLU(),nn.Dropout(0.4),nn.Linear(64,1))
    def forward(self,x):
        emb=self.embed(x); _,(hn,_)=self.lstm(emb)
        return self.fc(torch.cat([hn[-2],hn[-1]],dim=1)).squeeze(1)

model=SentimentLSTM().to(DEVICE)
criterion=nn.BCEWithLogitsLoss()
optimizer=optim.Adam(model.parameters(),lr=LR)
scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=2,factor=0.5)

def evaluate(loader):
    model.eval(); losses,preds,tgts=[],[],[]
    with torch.no_grad():
        for X,y in loader:
            X,y=X.to(DEVICE),y.to(DEVICE); logits=model(X)
            losses.append(criterion(logits,y).item())
            preds.extend(torch.sigmoid(logits).cpu().numpy()); tgts.extend(y.cpu().numpy())
    preds,tgts=np.array(preds),np.array(tgts)
    return np.mean(losses),((preds>=0.5)==tgts).mean(),preds,tgts

history={"tl":[],"vl":[],"ta":[],"va":[]}
best_acc,best_state=0,None
for epoch in range(1,EPOCHS+1):
    model.train(); tl,tc,tt=0,0,0
    for X,y in train_loader:
        X,y=X.to(DEVICE),y.to(DEVICE); optimizer.zero_grad(); logits=model(X)
        loss=criterion(logits,y); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        tl+=loss.item(); tc+=((torch.sigmoid(logits)>=0.5)==y).sum().item(); tt+=y.size(0)
    tl/=len(train_loader); ta=tc/tt; vl,va,vp,vt=evaluate(test_loader); scheduler.step(vl)
    history["tl"].append(tl); history["vl"].append(vl); history["ta"].append(ta); history["va"].append(va)
    print(f"Epoch {epoch}: train_acc={ta:.4f} val_acc={va:.4f}")
    if va>best_acc: best_acc=va; best_state={k:v.clone() for k,v in model.state_dict().items()}

model.load_state_dict(best_state)
_,acc,proba,true=evaluate(test_loader)
pred=(proba>=0.5).astype(int)
print(f"\nBest Test Acc: {acc:.4f}  AUC: {roc_auc_score(true,proba):.4f}")
print(classification_report(true,pred,target_names=["Negative","Positive"]))
torch.save(model.state_dict(),"sentiment_lstm.pth")
print("\n✅ Done!")
