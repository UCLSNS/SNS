import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 加载数据
df = pd.read_csv('beijing.csv')

df['TIME'] = pd.to_datetime(df['TIME'], format='%Y-%m-%d')
df = df.set_index('TIME')

data = pd.DataFrame({'TEMP_MAX': df['TEMPERATURE_max'], 'TEMP_MIN': df['TEMPERATURE_min'], 'RAIN': (df['PRECIPITATION'] > 0).astype(int)},
                    index = df.index, 
                    columns = ['TEMP_MAX', 'TEMP_MIN', 'RAIN'])

# 重采样和计算每日平均温度
#data = data.resample('D').mean()

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
                                                                                                                                                                                                                                                                                                      
# 构造数据集
X = []
Y_temp_max = []
Y_temp_min = []
Y_rain = []

look_back = 60

for i in range(len(data_scaled) - look_back - 1):
    X.append(data_scaled[i:(i + look_back), :])
    Y_temp_max.append(data_scaled[i + look_back, 0])
    Y_temp_min.append(data_scaled[i + look_back, 1])
    Y_rain.append(data_scaled[i + look_back, 2])

X = np.array(X)
Y_temp_max = np.array(Y_temp_max)
Y_temp_min = np.array(Y_temp_min)
Y_rain = np.array(Y_rain)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
trainX, testX = X[:train_size], X[train_size:]
trainY_temp_max, testY_temp_max = Y_temp_max[:train_size], Y_temp_max[train_size:]
trainY_temp_min, testY_temp_min = Y_temp_min[:train_size], Y_temp_min[train_size:]
trainY_rain, testY_rain = Y_rain[:train_size], Y_rain[train_size:]

# 构建和训练LSTM模型
model = Sequential()
model.add(LSTM(50, input_shape = (trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.2))
# 输出层
model.add(Dense(3))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(trainX, [trainY_temp_max, trainY_temp_min, trainY_rain], 
          epochs = 30,
          batch_size = 64,
          validation_data = (testX, [testY_temp_max, testY_temp_min, testY_rain]),
          verbose = 2)

# 预测未来温度
#last_batch = temperature_scaled[-look_back:]
#last_batch = last_batch.reshape((1, look_back, 1))
#next_temperature = model.predict(last_batch)
#next_temperature = scaler.inverse_transform(next_temperature)
#print('Predicted temperature for tomorrow:', next_temperature)

# 预测未来三天的温度和下雨
future_temp_max = []
future_temp_min = []
future_rain = []
last_batch = data_scaled[-look_back:]
last_batch = last_batch.reshape((1, look_back, 3))

# 循环三次来预测接下来的三天
for i in range(3):  # 预测未来三天
    next_temperature_max = model.predict(last_batch)[0][0]
    next_temperature_min = model.predict(last_batch)[0][1]
    next_rain = model.predict(last_batch)[0][2]

    # 反归一化温度
    temp_max = scaler.inverse_transform([[next_temperature_max, 0, 0]])[0][0]
    future_temp_max.append(temp_max)
    temp_min = scaler.inverse_transform([[0, next_temperature_min, 0]])[0][1]
    future_temp_min.append(temp_min)
    # 判断是否下雨
    future_rain.append(next_rain > 1)  
    new_batch = np.append(last_batch[:, 1:, :], [[[next_temperature_max, next_temperature_min, next_rain]]], axis=1)
    last_batch = new_batch.reshape((1, look_back, 3))


# 打印预测的温度和是否下雨
for i in range(len(future_temp_max)):  # future_temp和future_rain长度相同
    temp_Max = future_temp_max[i]
    temp_Min = future_temp_min[i]
    rain = future_rain[i]
    rain_status = "Rain" if rain else "No Rain"
    print('Day {}: Predicted max temperature: {}, Predicted min temperature: {}, Rain status: {}'.format(i + 1, temp_Max, temp_Min, rain_status))



