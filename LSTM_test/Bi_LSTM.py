import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation,Embedding,Bidirectional,Dropout,TimeDistributed,BatchNormalization
from keras.preprocessing import sequence

scaler_minmax = MinMaxScaler()

def load_data(file_name1,file_name2):
    df1 = pd.read_csv(file_name1, sep=',',header=None)
    df2 = pd.read_csv(file_name2, sep=',',header=None)
    data1 = np.array(df1).astype(float)
    data2 = np.array(df2).astype(float)
    data1to1 = scaler_minmax.fit_transform(data1)
    data2to1 = scaler_minmax.fit_transform(data2)
    train_x = data1to1[: 91960]
    test_x = data1to1[91960:100960]
    train_y = data2to1[: 9196]
    test_y = data2to1[9196:10096]
    # aa = np.reshape(train_x, (155, 200, 6))

    # print(aa)
    return train_x, train_y, test_x, test_y

def build_model():
    #train_x的维度为(n_samples, time_steps, input_dim) 样本总数，步长，维度
    model = Sequential()
    model.add(Bidirectional(LSTM(units=96, return_sequences=True), input_shape=(10, 3)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(units=96, return_sequences=False)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    model.summary();
    return model

def train_model(train_x, train_y, test_x, test_y):
    model = build_model()

    try:
        history =model.fit(train_x, train_y, batch_size=64, epochs=100, validation_data=(test_x, test_y))
        history_dict = history.history
        loss_value = history_dict["loss"]
        val_loss_value = history_dict["val_loss"]
        epochs = range(1, len(loss_value) + 1)
        plt.plot(epochs, loss_value, label ="Training loss")
        plt.plot(epochs, val_loss_value,'r:', label ="Validation loss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.show()
        # predict = model.predict(test_x)
        predict = scaler_minmax.inverse_transform(model.predict(test_x))
        # predict = np.reshape(predict, (33,1,2 ))
    except KeyboardInterrupt:
        print(predict)
        print(test_y)
    print("predict={},test_y={}".format(predict,test_y))
    # inverse_transform获得归一化前的原始数据
    plt.plot(scaler_minmax.inverse_transform(test_y), label='true data')
    plt.plot(predict, 'r:', label='predict')
    plt.legend()
    plt.show()


    return predict, test_y


if __name__ == '__main__':

    train_x, train_y, test_x, test_y = load_data("../resources/train_data/acc.csv","../resources/train_data/dis_vi.csv")
    train_x = np.reshape(train_x, (9196, 10, 3))
    test_x = np.reshape(test_x, (900, 10, 3))
    print("train_x.shape={},test_x.shape={}".format(train_x.shape, test_x.shape))
    predict_y, test_y = train_model(train_x, train_y, test_x, test_y)

