import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy.random import seed
import random
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import zscore
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
# from final_version.proposed_model import *

# 시드 값 설정
seed = 10
# 기본 시드 고정
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# CUDA 사용 시 추가 설정
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티-GPU 사용 시
    # CuDNN 결정론적 및 비결정론적 동작 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 데이터 준비
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
# 해석하는 파일명
fl = '13'

def plot(df, name, ylimit=None):
    ax = df.plot(figsize=(18,6), xlabel="Date", ylabel="VOLT/CURR", ylim=ylimit, grid=True, marker='o', ms=2) 
    # 범례 위치 설정 
    ax.legend(loc='upper left')
    plt.savefig('./figure/0910/CVAE/%s.png'%name, dpi=600)
    plt.clf() # figure 초기화
    plt.cla() # figure 축 초기화
    plt.close() # 현재 figure 닫기

raw = pd.read_csv(f"./AD-DTQ/{fl}.csv")
df = raw.set_index(raw.columns[0])
print('Resistance data', df.shape, '\n')
stat_ = df.describe(percentiles=None)
stat = stat_.apply(lambda x: x.apply(lambda y: f'{y:.2e}'))
plot(df, 'Resistance data',(0,0.2))

# 날짜 범위 설정
start_date = '2023-09-01 00:00:00'
end_date = '2023-09-14 23:59:59'

filtered_df = df.loc[start_date:end_date]

plot(filtered_df, '2 weeks filtered data')

# 이상이 있는 데이터의 넘버 입력 및 그래프 보여주기
data_err = 13
plot(filtered_df[f"R:QMP{data_err}"], 'filter data R:QMP-13 plot')

# 기준값(Threshold) 설정
threshold = 0.06

below_threshold = filtered_df[f"R:QMP{data_err}"] < threshold
times = pd.to_datetime(filtered_df.index[below_threshold])
time_diffs = times.to_series().diff()
groups = (time_diffs > pd.Timedelta(seconds=10)).cumsum()
time_ranges = times.to_series().groupby(groups).agg(['first','last'])

# 시작과 종료 시간을 저장할 리스트
start_times = []
end_times = []

for _,row in time_ranges.iterrows():
    start = row['first'].strftime('%Y-%m-%d %H:%M:%S')
    end = row['last'].strftime('%Y-%m-%d %H:%M:%S')
    start_times.append(start)
    end_times.append(end)

for i, (start, end) in enumerate(zip(start_times, end_times), 1):
    print(f'start {i-1} : {start}, end {i-1} : {end}')

# 각각의 데이터 프레임을 저장할 딕셔너리
dfs = {}

# 첫 번째 시작점부터 첫 종료지점까지의 데이터 프레임을 생성
dfs['df1'] = filtered_df[:start_times[0]]

# 각각의 범위에 해당하는 데이터 프레임을 생성
for i in range(len(start_times)):
    if i == 0:
        dfs[f'df{i}'] = filtered_df[:start_times[i]]
        continue
    
    dfs[f'df{i}'] = filtered_df[end_times[i-1]:start_times[i]]

    if i == len(start_times)-1:
        dfs[f'df{i+1}'] = filtered_df[end_times[i]:]

for i in range(len(dfs)):
    print(f'df{i}', 'has', dfs[f'df{i}'].shape, 'shape')

# 각 dataFrame마다 미세조정 실행
fix_dfs = {}

# 이상 전 데이터 넘버 입력
number= 1
# 이상 데이터 넘버 입력 
number_abnormal = 2

# 제거할 값 입력, 이상 데이터는 그대로 출력
del_front = 0
del_end = 10
k = 1
for i in [number]:
    fix_df = dfs[f'df{i}']
    fix_dfs[f'{k}'] = fix_df.loc[fix_df.index[del_front]:fix_df.index[len(fix_df)-del_end]]
    plot(fix_dfs[f'{k}'], 'normal event')
    k += 1

fix_dfs[f'{k}'] = dfs[f'df{number_abnormal}']
plot(fix_dfs[f'{k}'], 'abnormal event')

for i in range(1, len(fix_dfs)+1):
    print(f'fix_dfs{i}', 'has', fix_dfs[f'{i}'].shape, 'shape')

####################################################
### 2. Conditional Variational Auto-Encoder 학습 ###
####################################################

# 모델 파라미터
nb_epochs = 100
batch_size = 60
val_split = 0.1

threshold_result = 0.1

N_train = 1 # 정상데이터
N_test = 2 # 비정상데이터

train, test = fix_dfs[f'{N_train}'], fix_dfs[f'{N_test}']

# normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train)
X_test = scaler.transform(test)


# reshape inputs for LSTM [samples, timesteps, features]
X_train_all = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
print("Training data number", f'{N_train}', '&', "data shape:", X_train_all.shape)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
print("Test data number", f'{N_test}', '&', "data shape:", X_test.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder 정의
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim, dropout_prob=0.2):
        super(Encoder, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_mu = nn.Linear(128 + condition_dim, latent_dim)  # Latent mean
        self.fc_logvar = nn.Linear(128 + condition_dim, latent_dim)  # Latent log variance
        self.dropout = nn.Dropout(dropout_prob)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x, c):
        x = self.flatten(x)  # [batch_size, 1, 24] -> [batch_size, 24]
        x = F.gelu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)

        x = F.gelu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)

        # Condition을 Concatenate
        x = torch.cat([x, c], dim=-1)  # [batch_size, 128 + condition_dim]

        mu = self.fc_mu(x)  # Latent mean
        logvar = self.fc_logvar(x)  # Latent log variance

        return mu, logvar

# Decoder 정의
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, condition_dim, dropout_prob=0.2):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + condition_dim, 512)
        self.fc2 = nn.Linear(512, 128 * hidden_dim)
        self.fc_output = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.bn1 = nn.BatchNorm1d(128 * hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, z, c):
        z = torch.cat([z, c], dim=-1)  # [batch_size, latent_dim + condition_dim]

        h = F.gelu(self.fc1(z))
        h = F.gelu(self.fc2(h))
        h = self.bn1(h)
        h = h.view(h.size(0), self.hidden_dim, 128)  # [batch_size, 1, 128]

        # 출력층
        x_recon = torch.sigmoid(self.fc_output(h))

        return x_recon

# CVAE 모델 정의
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, condition_dim, dropout_prob=0.2):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, condition_dim, dropout_prob)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, condition_dim, dropout_prob)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        # Encode
        mu, logvar = self.encoder(x, c)
        # Reparameterization trick
        z = self.reparameterize(mu, logvar)
        # Decode
        x_recon = self.decoder(z, c)

        return x_recon, mu, logvar

# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 모델 및 학습파라미터 설정
input_dim = 24 # default : 24 features, 채널 수 
hidden_dim = 1 # default : 1[s]
latent_dim = 128 # default : 512
condition_dim = 14 # 총 14가지의 Case에 대해서 합산한 모델을 생성함.
dropout = 0.2

# 조건부-번이 오토인코더 모델 정의
model = LSTMCVAE(input_dim, latent_dim, hidden_dim, condition_dim, dropout).to(device)
# 옵티마이저 정의 : Adam, 학습률 : 10^-3
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 손실함수 정의 (MSE(reduction='mean') + KLD)
def loss_function(x_recon, x, mu, logvar):    

    MSE_lib = nn.functional.mse_loss(x_recon, x, reduction='mean') # divided into batch size, time steps
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)# see Appendix B from VAE paper:
                                                                                           # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    
    return MSE_lib + 1.0*KLD + 1e-12

X_train = np.array(X_train_all)
X_train = X_train_all.astype(np.float32)
# 학습 및 검증 데이터 분할 비율 = 9 : 1
X_train = X_train_all[:int(len(X_train_all)*0.9),:,:]
X_val = X_train_all[int(len(X_train_all)*0.9):,:,:]
X_test = np.array(X_test)
# tensor 위에 학습, 검증, 테스트셋 올리기
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
# 원-핫 벡터
one_hot_vector = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
# 원-핫 벡터를 샘플 수 만큼 늘리기
labels_train = np.tile(one_hot_vector, (len(X_train), 1))
labels_val = np.tile(one_hot_vector, (len(X_val), 1))
labels_test = np.tile(one_hot_vector, (len(X_test), 1))
# label을 tensor위에 올리기
labels_train = torch.tensor(labels_train, dtype=torch.float32)
labels_val = torch.tensor(labels_val, dtype=torch.float32)
labels_test = torch.tensor(labels_test, dtype=torch.float32)

# 학습용 데이터셋 및 데이터 로드 설정
dataset_train = CustomDataset(X_train, labels_train)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True) # batch_size default : 60, 1분 단위로 학습하기 위해서
# 검증용 데이터셋 및 데이터 로드 설정
dataset_val = CustomDataset(X_val, labels_val)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True) # batch_size default : 60
# 테스트용 데이터셋 및 데이터로드 설정
dataset_test = CustomDataset(X_test, labels_test)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True) # batch_size default : 60

train_losses = []
val_losses = []
for epoch in range(nb_epochs):
    model.train()
    train_loss = 0.0
    pbar_train = tqdm(dataloader_train, desc=f'Epoch {epoch + 1}/{nb_epochs}', total=len(dataloader_train), ncols=100)
    # train loss
    for data, label in pbar_train:
        x = data.to(device)
        cond = label.to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x, cond)
        loss = loss_function(x_recon=x_recon, x=x, mu=mu, logvar=logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # validation loss
    model.eval()
    val_loss = 0.0
    pbar_val = tqdm(dataloader_val, desc=f'Epoch {epoch + 1}/{nb_epochs}', total=len(dataloader_val), ncols=100)
    with torch.no_grad():
        for data, label in pbar_val:
            x = data.to(device)
            cond = label.to(device)
            x_recon, mu, logvar = model(x, cond)
            loss = loss_function(x_recon=x_recon, x=x, mu=mu, logvar=logvar)
            val_loss += loss.item()\
    
    # 평균 손실 저장
    train_losses.append(train_loss / len(dataloader_train))
    val_losses.append(val_loss / len(dataloader_val))

# Plot Training and Validation Loss      
fig, ax = plt.subplots(figsize=(18, 6), dpi=80)
ax.plot(train_losses, 'b', label='Train', linewidth=2)
ax.plot(val_losses, 'r', label='Validation', linewidth=2)
ax.set_title('Model loss')
ax.set_ylabel('Loss (mse)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.savefig('./figure/0910/CVAE/Model loss.png', dpi=600)

# plot the loss distribution of the training set
fig, ax = plt.subplots(figsize=(14, 6))
model.eval()
with torch.no_grad():
    # GPU에 올린 데이터를 CPU로 이동한 후 numpy()로 변환해야 함
    X_train = X_train.to(device)
    labels_train = labels_train.to(device)
    
    # 모델에서 예측 수행
    X_pred_train, _, _ = model(X_train, labels_train)  # 모델의 예측 수행
    X_pred_train = X_pred_train.cpu().numpy()  # 텐서를 CPU로 이동한 후 NumPy 배열로 변환
    
    # X_pred_train을 적절하게 reshape (확실히 필요한 크기로 변환)
    X_pred_train = X_pred_train.reshape(X_pred_train.shape[0], X_pred_train.shape[-1])
    
    # Pandas 데이터프레임으로 변환
    X_pred_train = pd.DataFrame(X_pred_train, columns=train.columns)
    # train의 인덱스를 예측된 데이터의 길이에 맞게 잘라줌
    if len(X_pred_train) < len(train):
        train = train.iloc[:len(X_pred_train)]  
    X_pred_train.index = train.index
    
    # 원래 학습 데이터 (X_train)을 NumPy로 변환
    Xtrain = X_train.cpu().numpy().reshape(X_train.shape[0], X_train.shape[-1])
    
    # 손실(MSE) 계산
    scored = pd.DataFrame(index=train.index)
    scored['Loss_mse'] = np.mean(np.abs(X_pred_train - Xtrain), axis=1)

    # 손실 분포 플로팅
    sns.histplot(scored['Loss_mse'], bins=20, kde=True, color='blue', ax=ax)
    ax.set_title('Loss Distribution')
    ax.set_xlim([0.0, .2])

    # 플롯 저장
    plt.savefig('./figure/0910/CVAE/Loss Distribution.png', dpi=600)
    plt.close()


# Loss graph
fig, ax = plt.subplots(figsize=(14, 6))
model.eval()
with torch.no_grad():
    X_test = X_test.to(device)
    labels_test = labels_test.to(device)

    X_pred,_,_ = model(X_test,labels_test)
    X_pred = X_pred.cpu().numpy()
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred = pd.DataFrame(X_pred, columns=test.columns)
    X_pred.index = test.index

    scored = pd.DataFrame(index=test.index)
    Xtest = X_test.cpu().numpy().reshape(X_test.shape[0], X_test.shape[2])
    scored['Loss_mse'] = np.mean(np.abs(X_pred - Xtest), axis=1)
    scored['Threshold'] = threshold_result
    scored['Anomaly'] = scored['Loss_mse'] > scored['Threshold']

# calculate the same metrics for the training set 
# and merge all data in a single dataframe for plotting
model.eval()
with torch.no_grad():
    # GPU에 올린 데이터를 CPU로 이동한 후 numpy()로 변환해야 함
    X_train = X_train.to(device)
    labels_train = labels_train.to(device)
    
    # 모델에서 예측 수행
    X_pred_train, _, _ = model(X_train, labels_train)  # 모델의 예측 수행
    X_pred_train = X_pred_train.cpu().numpy()  # 텐서를 CPU로 이동한 후 NumPy 배열로 변환
    
    # X_pred_train을 적절하게 reshape (확실히 필요한 크기로 변환)
    X_pred_train = X_pred_train.reshape(X_pred_train.shape[0], X_pred_train.shape[-1])
    
    # Pandas 데이터프레임으로 변환
    X_pred_train = pd.DataFrame(X_pred_train, columns=train.columns)
    # train의 인덱스를 예측된 데이터의 길이에 맞게 잘라줌
    if len(X_pred_train) < len(train):
        train = train.iloc[:len(X_pred_train)]  
    X_pred_train.index = train.index

    scored_train = pd.DataFrame(index=train.index)
    scored_train['Loss_mse'] = np.mean(np.abs(X_pred_train - Xtrain), axis=1)
    scored_train['Threshold'] = threshold_result
    scored_train['Anomaly'] = scored_train['Loss_mse'] > scored_train['Threshold']
    scored = pd.concat([scored_train, scored])

# 'Loss_mse' 컬럼에 대해 10개의 데이터 포인트에 대한 rolling mean 계산
scored['Loss_mse_smoothed'] = scored['Loss_mse'].rolling(window=2160).mean()
# 원본 scored 데이터프레임에서 'Loss_mse'와 'Threshold' 컬럼만 플로팅
scored[['Loss_mse', 'Threshold']].plot(ylim=[0, 0.5], color=['blue', 'red'], figsize=(14, 6), ax=ax)  # figsize 추가
# rolling mean을 가진 'Loss_mse_smoothed' 컬럼 추가하여 플로팅
scored['Loss_mse_smoothed'].plot(ax=ax, color='orange', linewidth=2, legend=True)
# 범례 추가
ax.legend(["Loss_mse", "Threshold", "Loss_mse Smoothed"])
ax.set_title('Loss graph')
ax.set_ylabel('Loss (mse)')
ax.set_xlabel('Time')
# 두 번째 플롯 표시
plt.savefig('./figure/0910/CVAE/Loss graph.png', dpi=600)
scored_reset = scored.reset_index(drop=True)



# scored_reset 데이터를 기반으로 새로운 인덱스 계산
new_index = scored_reset.index - max(scored_reset.index)

# 새로운 인덱스로 DataFrame 업데이트
scored_reset.index = new_index

fig, ax = plt.subplots(figsize=(14, 6))
scored_reset['Loss_mse_smoothed'].plot(ax=ax, color='orange', linewidth=2, legend=True)
ax.set_title('New Score Based Loss graph')
ax.set_ylabel('Loss (mse)')
ax.set_xlabel('Time')
plt.savefig('./figure/0910/CVAE/New Score Based Loss graph.png', dpi=600)
