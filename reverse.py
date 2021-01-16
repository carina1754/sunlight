#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import os
import random
import torch
import warnings
import pickle

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pytorch_lightning as pl


# In[2]:


warnings.filterwarnings("ignore")


# In[3]:


MAX_SEQ = 336


# In[4]:


FEATURES = ["Hour_Minute", "DHI", "DNI", "WS", "RH", "T"]
TARGET = "TARGET"


# In[5]:


OUTPUT_DIR = "./result"


# # Train Valid split

# In[6]:


def gen_range_map(min_val, max_val, step_size, init=None):
    if init:
        dic = init
    else:
        dic = {}
    for i in range((max_val - min_val) // step_size):
        dic[range(min_val + i * step_size, min_val + (i + 1) * step_size)] = (min_val + i * step_size + step_size // 2)
    return dic


# In[7]:


gen_range_map(10, 30, 5)


# In[8]:


def preprocess(df):
    df["Hour_Minute"] = df["Hour"] * 2 + df["Minute"] // 30
    df["ordering"] = (df["Day"] // 9)
    df["Day_"] = df["Day"] % 9
    df["RESPONSE"] = df[TARGET].astype("int32")
    df["DHI"] = df["DHI"].astype("int32")
    df["DNI"] = df["DNI"].astype("int32").clip(700, 1050)
    df["RH"] = (df["RH"] * 100).astype("int32")
    df["WS"] = (df["WS"] * 10).astype("int32")
    df["T"] = df["T"].astype("int32").clip(-20, 35)

    return df


# In[9]:


def range_mapping(df):
    dhi_dic = gen_range_map(1, 550, 10, {range(0,1): 0})
    dni_dic = gen_range_map(1, 1050, 10, {range(0,1): 0})
    ws_dic = gen_range_map(10, 120, 5)
    rh_dic = gen_range_map(700, 10000, 100)
    t_dic = gen_range_map(-20, 35, 5)
    df['DHI'] = df['DHI'].apply(lambda x: next((v for k, v in dhi_dic.items() if x in k), 0))
    df['DNI'] = df['DNI'].apply(lambda x: next((v for k, v in dni_dic.items() if x in k), 0))
    df['RH'] = df['RH'].apply(lambda x: next((v for k, v in rh_dic.items() if x in k), 0))
    df['WS'] = df['WS'].apply(lambda x: next((v for k, v in ws_dic.items() if x in k), 0))
    df['T'] = df['T'].apply(lambda x: next((v for k, v in t_dic.items() if x in k), 0))
    return df


# In[10]:


def load_train_df(path):
    df = pd.read_csv(path)
    df = preprocess(df)
    df = range_mapping(df)
    label_encoder_dict = {}
    for col in FEATURES + ["RESPONSE"]:
        le = LabelEncoder()
        le = le.fit(df[col])
        df[col] = le.transform(df[col])
        label_encoder_dict[col] = le
    return df, label_encoder_dict


# In[11]:


def load_test_df(path, label_encoder_dict):
    dfs = []
    for idx, filename in enumerate(sorted(os.listdir(path))):
        if ".csv" in filename:
            df = pd.read_csv(f"{path}/{filename}")
            df = preprocess(df)
            df = range_mapping(df)
            for col in FEATURES+["RESPONSE"]:
                df[col] = label_encoder_dict[col].transform(df[col])
            df["ordering"] = filename
            dfs.append(df)
    return pd.concat(dfs)


# In[12]:


n_x = 96
n_hm = 48
n_dhi = 54
n_dni = 36
n_ws = 23
n_rh = 94
n_t = 12
n_target = 100


# In[13]:


def get_time_from_int(i):
    i = i % 48
    if i % 2 == 0:
        m = "00"
    else:
        m = "30"
        
    return f"{i//2}h{m}m"


# In[14]:


def make_submit_df(outputs, quantile):
    output = outputs[0]
    indexes = []
    values = []
    for filename, preds in zip(output["filename"], output["output"]):
        for idx, pred in enumerate(preds):
            hm = get_time_from_int(idx)
            if idx < 48:
                index = f"{filename}_Day7_{hm}"
            else:
                index = f"{filename}_Day8_{hm}"
            
            indexes.append(index)
            values.append(pred.item())
    res = pd.DataFrame({
        "index": indexes,
        f"{quantile}": values
    })
    res.index = res.pop("index")
    return res


# # Dataset

# In[15]:


class SASEFDataset(Dataset):
    def __init__(self, group, max_seq=MAX_SEQ, test=False):
        super(SASEFDataset, self).__init__()
        self.samples = {}
        self.max_seq = max_seq
        self.test = test
        
        self.orderings = []
        
        for i, ordering in enumerate(group.index):
            if test:
                hm, dhi, dni, ws, rh, t, r = group[ordering]
                if len(hm) < max_seq:
                    continue
                self.orderings.append(ordering)
                self.samples[ordering] = (hm, dhi, dni, ws, rh, t, r)
            else:
                hm, dhi, dni, ws, rh, t, r, target = group[ordering]
                index = f"{ordering}"
                if len(hm) < max_seq:
                    continue
                self.orderings.append(index)
                self.samples[index] = (hm, dhi, dni, ws, rh, t, r, target)
    
    def __len__(self):
        return len(self.orderings)
    
    def __getitem__(self, index):
        ordering = self.orderings[index]
        if self.test:
            hm, dhi, dni, ws, rh, t, r = self.samples[ordering]
            x = np.array([i for i in range(48)] * 2, dtype=int)
            return ordering, x, r, hm, dhi, dni, ws, rh, t
        else:
            hm, dhi, dni, ws, rh, t, r, target = self.samples[ordering]
            x = np.array([i for i in range(48)] * 2, dtype=int)
            return x, r[:MAX_SEQ], hm[:MAX_SEQ], dhi[:MAX_SEQ], dni[:MAX_SEQ], ws[:MAX_SEQ], rh[:MAX_SEQ], t[:MAX_SEQ], target[MAX_SEQ:]


# In[16]:


train_df, label_encoder_dict = load_train_df("./data/train/train.csv")


# In[17]:


test_df = load_test_df("./data/test", label_encoder_dict)


# In[18]:


test_df = load_test_df("./data/test", label_encoder_dict)
test_group = test_df[FEATURES + [TARGET] + ["ordering" , "RESPONSE"]].groupby("ordering").apply(lambda r: (
    r["Hour_Minute"].values, 
    r["DHI"].values, 
    r["DNI"].values, 
    r["WS"].values, 
    r["RH"].values, 
    r["T"].values,
    r["RESPONSE"].values))
dataset = SASEFDataset(test_group, test=True)


# In[19]:


class SASEFDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        
    def setup(self, stage):
        if stage == "fit":
            train_df, label_encoder_dict = load_train_df("./data/train/train.csv")
            self.label_encoder_dict = label_encoder_dict
            train_group = train_df[FEATURES + [TARGET] + ["ordering" , "RESPONSE"]].groupby("ordering").apply(lambda r: (
                r["Hour_Minute"].values, 
                r["DHI"].values, 
                r["DNI"].values, 
                r["WS"].values, 
                r["RH"].values, 
                r["T"].values,
                r["RESPONSE"].values,
                r[TARGET].values))
            valid_ordering = random.sample(range(0, len(train_group)), int(len(train_group)*0.1))
            valid_group = train_group[train_group.index.isin(valid_ordering)]
            train_group = train_group.drop(valid_group.index)
            self.train_dataset = SASEFDataset(train_group)
            self.valid_dataset = SASEFDataset(valid_group)
        if stage == "test":
            test_df = load_test_df("./data/test", self.label_encoder_dict)
            test_group = test_df[FEATURES + [TARGET] + ["ordering" , "RESPONSE"]].groupby("ordering").apply(lambda r: (
                r["Hour_Minute"].values, 
                r["DHI"].values, 
                r["DNI"].values, 
                r["WS"].values, 
                r["RH"].values, 
                r["T"].values,
                r["RESPONSE"].values))
            self.test_dataset = SASEFDataset(test_group, test=True)
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=100)


# # Model

# In[20]:


def future_mask(seq_length):
    future_mask = (np.triu(np.ones([seq_length, seq_length]), k = 1)).astype('bool')
    return torch.from_numpy(future_mask)


# In[21]:


class FFN(nn.Module):
    def __init__(self, dropout = 0.1, state_size_in = 200, state_size_out = 200, forward_expansion = 1, bn_size=MAX_SEQ - 1):
        super(FFN, self).__init__()
        self.state_size_in = state_size_in
        self.state_size_out = state_size_out
        
        self.lr1 = nn.Linear(state_size_in, forward_expansion * state_size_in)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(forward_expansion * state_size_in, state_size_out)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.relu(self.lr1(x))
        x = self.lr2(x)
        return self.dropout(x)


# In[22]:


class Encoder(nn.Module):
    def __init__(self, n_x, max_seq, embed_dim, 
                 dropout, forward_expansion, num_layers, heads=8):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.pos_embedding = nn.Embedding(max_seq, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_normal = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        device = x.device
        pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)
        pos_x = self.pos_embedding(pos_id)
        x = self.dropout(self.layer_normal(x))
        return x


# In[23]:


class Decoder(nn.Module):
    def __init__(self, n_target, n_hm, n_dhi, n_dni, n_ws, n_rh, n_t,
                 max_seq, embed_dim, dropout, forward_expansion,
                 num_layers, heads=8):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.r_embedding = nn.Embedding(n_target + 1, embed_dim)
        self.dhi_embedding = nn.Embedding(n_dhi + 1, embed_dim)
        self.dni_embedding = nn.Embedding(n_dni + 1, embed_dim)
        self.ws_embedding = nn.Embedding(n_ws + 1, embed_dim)
        self.rh_embedding = nn.Embedding(n_rh + 1, embed_dim)
        self.t_embedding = nn.Embedding(n_t + 1, embed_dim)
        
        self.pos_embedding = nn.Embedding(max_seq, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.layer_normal = nn.LayerNorm(embed_dim)
        
    def forward(self, target, hm, dhi, dni, ws, rh, t):
        device = target.device
        target = self.r_embedding(target)
        dhi = self.dhi_embedding(dhi)
        dni = self.dni_embedding(dni)
        ws = self.ws_embedding(ws)
        rh = self.rh_embedding(rh)
        t = self.t_embedding(t)
        
        pos_id = torch.arange(target.size(1)).unsqueeze(0).to(device)
        pos_x = self.pos_embedding(pos_id)
        x = self.dropout(self.layer_normal(target + hm + dhi + dni + ws + rh + t))
        return x


# In[24]:


class SASEFModel(pl.LightningModule):
    def __init__(self, embed_dim, lr, optimizer, dropout, quantile, n_target=n_target, n_x=n_x, n_hm=n_hm, n_dhi=n_dhi, n_dni=n_dni, 
                 n_ws=n_ws, n_rh=n_rh, n_t=n_t,
                 max_seq=MAX_SEQ, 
                 enc_layers=2, dec_layers=2, forward_expansion=1, heads=8):
        super(SASEFModel, self).__init__()
        self.hm_embedding = nn.Embedding(48, embed_dim)
        self.encoder = Encoder(n_x, max_seq, 
                               embed_dim, dropout, 
                               forward_expansion, enc_layers, 
                               heads)
        self.decoder = Decoder(n_target, n_hm, n_dhi, n_dni, n_ws, 
                               n_rh, n_t, max_seq, embed_dim, dropout, 
                               forward_expansion, dec_layers, 
                               heads)
        self.transformer = nn.Transformer(embed_dim, heads, 
                                          enc_layers, dec_layers,
                                          embed_dim*forward_expansion, 
                                          dropout)
        self.ffn = FFN(dropout, embed_dim, embed_dim, forward_expansion = forward_expansion)
        self.output = FFN(dropout, max_seq, 96, forward_expansion = forward_expansion)
        self.pred = nn.Linear(embed_dim, 1)
        self.layer_normal = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
        
        self.lr = lr
        self.optimizer = optimizer
        self.quantile = quantile

    def forward(self, x, r, hm, dhi, dni, ws, rh, t):
        x = x.reshape(x.shape[0] * 2, 48)
        x = self.hm_embedding(x)
        x = x.reshape(x.shape[0] // 2, 48 * 2, self.embed_dim)
        ex = self.encoder(x)
        
        
        hm = hm.reshape(hm.shape[0] * 7, 48)
        hm = self.hm_embedding(hm)
        hm = hm.reshape(hm.shape[0] // 7, 48 * 7, self.embed_dim)
        dx = self.decoder(r, hm, dhi, dni, ws, rh, t)
        
        ex = ex.permute(1, 0, 2)
        dx = dx.permute(1, 0, 2)
        
        device = ex.device
        mask = future_mask(ex.size(0)).to(device)
        tgt_mask = future_mask(dx.size(0)).to(device)
        att_output = self.transformer(ex, dx, src_mask=mask, tgt_mask=tgt_mask)
        att_output = self.layer_normal(att_output)
        att_output = att_output.permute(1, 0, 2)
        
        x = self.ffn(att_output)
        x = self.dropout(self.layer_normal(x + att_output))
        
        x = self.pred(x)
        x = x.squeeze(-1)
        x = self.output(x)
        return x.squeeze(-1)
    
    def training_step(self, batch, batch_idx):
        x, r, hm, dhi, dni, ws, rh, t, label = batch
        label = label.float()
        output = self(x, r, hm, dhi, dni, ws, rh, t)
        loss = self.quantile_loss(output, label)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, r, hm, dhi, dni, ws, rh, t, label = batch
        label = label.float()
        output = self(x, r, hm, dhi, dni, ws, rh, t)
        loss = self.quantile_loss(output, label)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        filename, x, r, hm, dhi, dni, ws, rh, t = batch
        output = self(x, r, hm, dhi, dni, ws, rh, t)
        return {"filename": filename, "output": output}
    
    def test_epoch_end(self, outputs):
        make_submit_df(outputs, quantile).to_csv(f"{OUTPUT_DIR}/output{self.quantile}.csv")
    
    def configure_optimizers(self):
        if self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.optimizer == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.lr)
        
    def quantile_loss(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        errors = target - preds
        loss = torch.max((self.quantile - 1) * errors, self.quantile * errors).unsqueeze(1)
        loss = torch.mean(torch.sum(loss, dim=1))
        return loss


# In[25]:


def run(quantile):
    sasef_data = SASEFDataModule(batch_size=16)
    sasef_data.setup("fit")
    model = SASEFModel(256, 1e-3, 'adam', 0.1, quantile, enc_layers=10, dec_layers=10)
    early_stop_callback = EarlyStopping(
       monitor='val_loss',
       min_delta=0.00,
       patience=3,
       verbose=False,
       mode='min'
    )
    checkpoint_callback = ModelCheckpoint(f"{OUTPUT_DIR}/best{quantile}", save_top_k=1, monitor='val_loss', mode='min')
    trainer = pl.Trainer(callbacks=[early_stop_callback, checkpoint_callback], check_val_every_n_epoch=1, gpus=1, weights_summary=None)
    trainer.fit(model, sasef_data)
    
    sasef_data.setup("test")
    trainer.test(ckpt_path=f"{OUTPUT_DIR}/best{quantile}.ckpt")
    return trainer


# In[26]:


best_losses = []
for quantile in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    trainer = run(quantile)
    best_losses.append(trainer.logged_metrics["val_loss"])


# In[27]:


print(sum(best_losses) / len(best_losses))


# # Submit file gen

# In[28]:


df_submit = pd.concat([pd.read_csv(f"{OUTPUT_DIR}/output{quantile}.csv") for quantile in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]], axis=1)
df_submit["id"] = df_submit.pop("index").iloc[:, 0]
df_submit.index = df_submit.pop("id")
df_submit = df_submit.rename(columns={f"{quantile}":f"q_{quantile}" for quantile in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]})


# In[29]:


df_submit = df_submit.clip(0)


# In[30]:


df_submit.to_csv("./data/submission.csv")


# In[ ]:




