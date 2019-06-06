import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
scaler_minmax = MinMaxScaler()
def load_data(file_name1):
    df1 = pd.read_csv(file_name1, sep=',',header=None)

    # data1 = scaler_minmax.fit_transform(df1[0])
    data1 = np.array(df1[0]).astype(int)
    data2 = np.array(df1[1]).astype(int)
    data1to1 = scaler_minmax.fit_transform(data1.reshape(-1,1))
    data2to1 = scaler_minmax.fit_transform(data2.reshape(-1,1))

    train_x = data1to1[: 990]
    test_x = data1to1[990:1190]
    train_y = data2to1[: 990]
    test_y = data2[990:1190]

    return train_x, train_y, test_x, test_y

def build_model():
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(LSTM(input_shape=(None, 1) ,units=96, return_sequences=True))
    model.add(LSTM(96, return_sequences=False))
    print(model.layers)
    model.add(Dense(units=1))

    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='adam')

    return model

def train_model(train_x, train_y, test_x, test_y):
    model = build_model()

    try:
        model.fit(train_x, train_y, batch_size=32, epochs=100, validation_data=(test_x, test_y))
        # predict = model.predict(test_x)
        predict = scaler_minmax.inverse_transform(model.predict(test_x))
    except KeyboardInterrupt:
        print(predict)
        print(test_y)
    print("predict={},test_y={}".format(predict,test_y))

    # inverse_transform获得归一化前的原始数据
    plt.plot(test_y, label='true data')
    plt.plot(predict, 'r:', label='predict')
    plt.legend()
    plt.show()
    # print(test_y)

    return predict, test_y


if __name__ == '__main__':

    train_x, train_y, test_x, test_y = load_data("../../resources/train_data/test_train.csv")
    train_x = np.reshape(train_x, (990, 1, 1))
    test_x = np.reshape(test_x, (200, 1, 1))
    print("train_x.shape={},test_x.shape={}".format(train_x.shape, test_x.shape))
    predict_y, test_y = train_model(train_x, train_y, test_x, test_y)

