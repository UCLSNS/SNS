import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# 加载数据
df = pd.read_csv('AAPL_data.csv')

df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df = df.set_index('Date')

data = pd.DataFrame({'PRICE_MAX': df['High'], 'PRICE_MIN': df['Low']},
                    index = df.index, 
                    columns = ['PRICE_MAX', 'PRICE_MIN'])

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# 构造数据集
X = []
Y_price_max = []
Y_price_min = []

look_back = 5

for i in range(len(data_scaled) - look_back - 1):
    X.append(data_scaled[i:(i + look_back), :])
    Y_price_max.append(data_scaled[i + look_back, 0])
    Y_price_min.append(data_scaled[i + look_back, 1])

X = np.array(X)
Y_price_max = np.array(Y_price_max)
Y_price_min = np.array(Y_price_min)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
trainX, testX = X[:train_size], X[train_size:]
trainY_price_max, testY_price_max = Y_price_max[:train_size], Y_price_max[train_size:]
trainY_price_min, testY_price_min = Y_price_min[:train_size], Y_price_min[train_size:]

# 构建和训练LSTM模型
model = Sequential()
model.add(LSTM(30, input_shape = (trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(2))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# 早停机制
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model.fit(trainX, [trainY_price_max, trainY_price_min], 
          epochs = 30,
          batch_size = 32,  # 减小批次大小
          validation_data = (testX, [testY_price_max, testY_price_min]),
          verbose = 2,
          callbacks = [early_stopping])  # 使用早停机制

# 预测未来三天的最高价格和最低价格
future_price_max = []
future_price_min = []
last_batch = data_scaled[-look_back:]
last_batch = last_batch.reshape((1, look_back, 2))

# 循环三次来预测接下来的三天
for i in range(3):  # 预测未来三天
    next_price_max = model.predict(last_batch)[0][0]
    next_price_min = model.predict(last_batch)[0][1]

    # 反归一化
    price_max = scaler.inverse_transform([[next_price_max, 0]])[0][0]
    future_price_max.append(price_max)
    price_min = scaler.inverse_transform([[0, next_price_min]])[0][1]
    future_price_min.append(price_min) 
    new_batch = np.append(last_batch[:, 1:, :], [[[next_price_max, next_price_min]]], axis=1)
    last_batch = new_batch.reshape((1, look_back, 2))

# 打印预测的价格
for i in range(len(future_price_max)):  
    price_Max = future_price_max[i]
    price_Min = future_price_min[i]
    print('Day {}: Predicted highest price: {}, Predicted lowest price: {}'.format(i + 1, price_Max, price_Min))


