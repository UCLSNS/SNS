import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 加载数据
df = pd.read_csv('data.csv')

df['TIME'] = pd.to_datetime(df['TIME'], format='%Y%m%d%H')
df = df.set_index('TIME')

data = pd.DataFrame({'TEMPERATURE': df['TEMPERATURE [degC]'],'RAIN': (df['PRECIPITATION [mm/6hr]'] > 0).astype(int)},
                    index = df.index, 
                    columns = ['TEMPERATURE', 'RAIN'])

# 重采样和计算每日平均温度
data = data.resample('D').mean()

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# 构造数据集
X = []
Y_temp = []
Y_rain = []

look_back = 60

for i in range(len(data_scaled) - look_back - 1):
    X.append(data_scaled[i:(i + look_back), :])
    Y_temp.append(data_scaled[i + look_back, 0])
    Y_rain.append(data_scaled[i + look_back, 1])

X = np.array(X)
Y_temp = np.array(Y_temp)
Y_rain = np.array(Y_rain)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
trainX, testX = X[:train_size], X[train_size:]
trainY_temp, testY_temp = Y_temp[:train_size], Y_temp[train_size:]
trainY_rain, testY_rain = Y_rain[:train_size], Y_rain[train_size:]

# 构建和训练LSTM模型
model = Sequential()
model.add(LSTM(50, input_shape = (trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(2))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(trainX, [trainY_temp, trainY_rain], 
          epochs = 20,
          batch_size = 64,
          validation_data = (testX, [testY_temp, testY_rain]),
          verbose = 2)

# 预测未来温度
#last_batch = temperature_scaled[-look_back:]
#last_batch = last_batch.reshape((1, look_back, 1))
#next_temperature = model.predict(last_batch)
#next_temperature = scaler.inverse_transform(next_temperature)
#print('Predicted temperature for tomorrow:', next_temperature)

# 预测未来三天的温度和下雨
future_temp = []
future_rain = []
last_batch = data_scaled[-look_back:]
last_batch = last_batch.reshape((1, look_back, 2))

# 循环三次来预测接下来的三天
for i in range(3):  # 预测未来三天
    next_temperature = model.predict(last_batch)[0][0]
    next_rain = model.predict(last_batch)[0][1]
    # 反归一化温度
    temp = scaler.inverse_transform([[next_temperature, 0]])[0][0]
    future_temp.append(temp)
    # 判断是否下雨
    future_rain.append(next_rain > 0.5)  
    new_batch = np.append(last_batch[:, 1:, :], [[[next_temperature, next_rain]]], axis=1)
    last_batch = new_batch.reshape((1, look_back, 2))


# 打印预测的温度和是否下雨
for i in range(len(future_temp)):  # future_temp和future_rain长度相同
    temp = future_temp[i]
    rain = future_rain[i]
    rain_status = "Rain" if rain else "No Rain"
    print('Day {}: Predicted temperature: {}, Rain status: {}'.format(i + 1, temp, rain_status))



