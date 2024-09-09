import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy.random import seed
from keras.layers import Input,  Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
from scipy.stats import zscore

# set random seed
seed(10)

# 해석하는 파일명
fl = '13'

def plot(df, name, ylimit=None):
    ax = df.plot(figsize=(18,6), xlabel="Date", ylabel="VOLT/CURR", ylim=ylimit, grid=True, marker='o', ms=2) 
    # 범례 위치 설정 
    ax.legend(loc='upper left')
    plt.savefig('./figure/0909/LSTM/%s.png'%name, dpi=600)
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

#################################
### 2. LSTM Auto-Encoder 학습 ###
#################################

# 모델 정의
def autoencoder_model(X, Unit1, Unit2, Unit3, Unit4):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(Unit1, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(Unit2, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(Unit3, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(Unit4, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model

# 모델 파라미터
nb_epochs = 100
batch_size = 60
val_split = 0.1
Unit1, Unit2, Unit3, Unit4 = 128, 32, 32, 128
threshold_result = 0.1

N_train = 1 # 정상데이터
N_test = 2 # 비정상데이터

train, test = fix_dfs[f'{N_train}'], fix_dfs[f'{N_test}']

# normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train)
X_test = scaler.transform(test)

# reshape inputs for LSTM [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
print("Training data number", f'{N_train}', '&', "data shape:", X_train.shape)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
print("Test data number", f'{N_test}', '&', "data shape:", X_test.shape)

print(X_train.dtype)
# create the autoencoder model
model = autoencoder_model(X_train, Unit1, Unit2, Unit3, Unit4)
model.compile(optimizer='adam', loss='mse')
print(model.summary())

# fit the model to the data
history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size, validation_split=val_split).history
model.save_weights('./model/LSTM.weights.h5')
print("학습 모델 저장완료!")

# Plot Training and Validation Loss      
fig, ax = plt.subplots(figsize=(18, 6), dpi=80)
ax.plot(history['loss'], 'b', label='Train', linewidth=2)
ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
ax.set_title('Model loss')
ax.set_ylabel('Loss (mse)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.savefig('./Model loss.png', dpi=600)
# plot the loss distribution of the training set
fig, ax = plt.subplots(figsize=(14, 6))
X_pred = model.predict(X_train, verbose=0)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=train.columns)
X_pred.index = train.index

scored = pd.DataFrame(index=train.index)
Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
scored['Loss_mse'] = np.mean(np.abs(X_pred - Xtrain), axis=1)
sns.histplot(scored['Loss_mse'], bins=20, kde=True, color='blue', ax=ax)  # ax=ax를 추가합니다.
ax.set_title('Loss Distribution')
ax.set_xlim([0.0, .2])
plt.savefig('./Loss Distribution.png', dpi=600)  # 첫 번째 플롯 표시

# Loss graph
fig, ax = plt.subplots(figsize=(14, 6))
X_pred = model.predict(X_test, verbose=0)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=test.columns)
X_pred.index = test.index

scored = pd.DataFrame(index=test.index)
Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
scored['Loss_mse'] = np.mean(np.abs(X_pred - Xtest), axis=1)
scored['Threshold'] = threshold_result
scored['Anomaly'] = scored['Loss_mse'] > scored['Threshold']

# calculate the same metrics for the training set 
# and merge all data in a single dataframe for plotting
X_pred_train = model.predict(X_train, verbose=0)
X_pred_train = X_pred_train.reshape(X_pred_train.shape[0], X_pred_train.shape[2])
X_pred_train = pd.DataFrame(X_pred_train, columns=train.columns)
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
plt.savefig('./Loss graph.png', dpi=600)
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
plt.savefig('./New Score Based Loss graph.png', dpi=600)
