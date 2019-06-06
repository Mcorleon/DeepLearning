import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation


def build_model():
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(LSTM(input_shape=(None, 1) ,units=50, return_sequences=False))
    # model.add(LSTM(100, return_sequences=False))
    print(model.layers)
    model.add(Dense(units=1))

    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='adam')

    return model

def train_model():
    df = pd.read_csv("../../resources/train_data/international-airline-passengers.csv", usecols=['passengers'])
    scaler_minmax = MinMaxScaler()
    data = scaler_minmax.fit_transform(df)
    infer_seq_length = 10  # 用于推断的历史序列长度
    d = []
    for i in range(data.shape[0] - infer_seq_length):
        d.append(data[i:i + infer_seq_length + 1].tolist())
    d = np.array(d)
    split_rate = 0.9
    train_x, train_y = d[:int(d.shape[0] * split_rate), :-1], d[:int(d.shape[0] * split_rate), -1]

    print(train_y)
    # model = build_model()

    # try:
    #     model.fit(train_x, train_y, batch_size=32, epochs=100, validation_split=0.1)
    #     predict = scaler_minmax.inverse_transform(model.predict(d[:, :-1]))
    #
    # except KeyboardInterrupt:
    #     print(predict)
    # print(predict)


if __name__ == '__main__':
    train_model()


