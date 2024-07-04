# 載入相關套件
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 載入測試資料 -- 亞馬遜
df = pd.read_csv('./RNN/AMZN_2006-01-01_to_2018-01-01.csv', 
                 index_col='Date', parse_dates=['Date'])
df.head()

df.tail()

# 只使用收盤價
df = df['Close']

# 繪圖
plt.figure(figsize = (12, 6))
plt.plot(df, label='Stock Price')
plt.legend(loc='best')
plt.show()

# 參數設定
look_back = 40    # 以過去 40 期為特徵(X)
forward_days = 10 # 一次預測 10 天 (y)
num_periods = 20  # 測試資料量設定 20 期

# 特徵常態化
from sklearn.preprocessing import MinMaxScaler
scl = MinMaxScaler()
array = df.values.reshape(df.shape[0],1)
array = scl.fit_transform(array)

# 前置處理函數，取得模型輸入的格式
# look_back：特徵(X)個數，forward_days：目標(y)個數，jump：移動視窗
def processData(data, look_back, forward_days,jump=1):
    X,Y = [],[]
    for i in range(0,len(data) -look_back -forward_days +1, jump):
        X.append(data[i:(i+look_back)])
        Y.append(data[(i+look_back):(i+look_back+forward_days)])
    return np.array(X),np.array(Y)

# 資料切割成訓練資料及測試資料
# 一次預測 10 天，共 20 期
division = len(array) - num_periods*forward_days

# 再往前推 40 天當第一筆的 X
array_test = array[division-look_back:]
array_train = array[:division]

# 前置處理、資料切割
# 測試資料前置處理，注意最後一個參數，一次預測 10天，不重疊
X_test,y_test = processData(array_test,look_back,forward_days,forward_days)
y_test = np.array([list(a.ravel()) for a in y_test])

# 訓練資料前置處理
X,y = processData(array_train,look_back,forward_days)
y = np.array([list(a.ravel()) for a in y])

# 資料切割成訓練資料及驗證資料
from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20)

print(X_train.shape)
print(X_validate.shape)
print(X_test.shape)
print()
print(y_train.shape)
print(y_validate.shape)
print(y_test.shape)


NUM_NEURONS_FirstLayer = 50
NUM_NEURONS_SecondLayer = 30
EPOCHS = 10

# 建立模型
model = Sequential()
model.add(LSTM(NUM_NEURONS_FirstLayer,input_shape=(look_back,1), return_sequences=True))
model.add(LSTM(NUM_NEURONS_SecondLayer,input_shape=(NUM_NEURONS_FirstLayer,1)))
model.add(Dense(forward_days))

#設定Optimizer
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

# 訓練模型
history = model.fit(X_train,y_train,epochs=EPOCHS,validation_data=(X_validate,y_validate)
                    ,shuffle=True,batch_size=2, verbose=2)
# 繪製損失函數
plt.figure(figsize = (12, 6))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc='best')
plt.show()

model.save('./RNN/stock.h5')

# 前置處理、資料切割
# 測試資料前置處理，注意最後一個參數，一次預測 1天
X_test,y_test = processData(array_test,look_back,1,1)
y_test = np.array([list(a.ravel()) for a in y_test])
        
# 測試資料預測
Xt = model.predict(X_test)
print(Xt.shape)

Xt = Xt[:, 0]

# 繪製測試資料預測值
plt.figure(figsize = (12, 6))
# 繪製 1 條預測值，scl.inverse_transform：還原常態化
plt.plot(scl.inverse_transform(Xt.reshape(-1,1)), color='r', label='Prediction')

# 繪製實際值
plt.plot(scl.inverse_transform(y_test.reshape(-1,1)), label='Target')
plt.legend(loc='best')
plt.show()

# 測試資料前置處理，注意最後一個參數，一次預測 10天，移動視窗不重疊
X_test,y_test = processData(array_test,look_back,forward_days,forward_days)
y_test = np.array([list(a.ravel()) for a in y_test])

# 測試資料預測
Xt = model.predict(X_test)
Xt.shape

# 繪製測試資料預測值
plt.figure(figsize = (12, 6))
# 繪製 20 條預測值，scl.inverse_transform：還原常態化
for i in range(0,len(Xt)):
    plt.plot([x + i*forward_days for x in range(len(Xt[i]))], 
             scl.inverse_transform(Xt[i].reshape(-1,1)), color='r')

# 指定預測值 label    
plt.plot(0, scl.inverse_transform(Xt[i].reshape(-1,1))[0], color='r'
         , label='Prediction') 

# 繪製實際值
plt.plot(scl.inverse_transform(y_test.reshape(-1,1)), label='Target')
plt.legend(loc='best')
plt.show()

# 全部資料預測
division = len(array) - num_periods*forward_days
array_test = array[division-look_back:]

# 去掉不能整除的資料，取完整的訓練資料
leftover = division%forward_days+1
array_train = array[leftover:division]

Xtrain,ytrain = processData(array_train,look_back,forward_days,forward_days)
Xtest,ytest = processData(array_test,look_back,forward_days,forward_days)

# 預測
Xtrain = model.predict(Xtrain)
Xtrain = Xtrain.ravel() # 轉成一維

Xtest = model.predict(Xtest)
Xtest = Xtest.ravel() # 轉成一維

# 合併訓練資料與測試資料
y = np.concatenate((ytrain, ytest), axis=0)

# 繪製訓練資料預測值
plt.figure(figsize = (12, 6))
plt.plot([x for x in range(look_back+leftover, len(Xtrain)+look_back+leftover)], 
         scl.inverse_transform(Xtrain.reshape(-1,1)), color='r', label='Train')
# 繪製測試資料預測值
plt.plot([x for x in range(look_back +leftover+ len(Xtrain), 
         len(Xtrain)+len(Xtest)+look_back+leftover)], 
         scl.inverse_transform(Xtest.reshape(-1,1)), color='y', label='Test')

# 繪製實際值
plt.plot([x for x in range(look_back+leftover, 
                           look_back+leftover+len(Xtrain)+len(Xtest))], 
         scl.inverse_transform(y.reshape(-1,1)), color='b', label='Target')

plt.legend(loc='best')
plt.show()

# 繪製測試資料預測值
plt.figure(figsize = (12, 6))
# 全部連成一線
plt.plot(scl.inverse_transform(Xtest.reshape(-1,1)))
# 畫20條線
for i in range(0,len(Xt)):
    plt.plot([x + i*forward_days for x in range(len(Xt[i]))], 
             scl.inverse_transform(Xt[i].reshape(-1,1)), color='r')

# 改用 GRU 模型
from tensorflow.keras.layers import GRU

model_GRU = Sequential()
model_GRU.add(GRU(NUM_NEURONS_FirstLayer,input_shape=(look_back,1)
                  , return_sequences=True))
model_GRU.add(GRU(NUM_NEURONS_SecondLayer
                  ,input_shape=(NUM_NEURONS_FirstLayer,1)))
model_GRU.add(Dense(forward_days))
model_GRU.compile(loss='mean_squared_error', optimizer='adam')

history = model_GRU.fit(X_train,y_train,epochs=EPOCHS
                    ,validation_data=(X_validate,y_validate)
                    ,shuffle=True,batch_size=2, verbose=2)






