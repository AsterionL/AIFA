# coding=utf-8

from __future__ import print_function
from AI_Data.dataSource import DataSource
import numpy as np
import tflearn
from tflearn.data_utils import to_categorical
from multiprocessing import Process

data_source = DataSource()


wind_code_list= ["600000.SH","600016.SH","600019.SH",
                 "600028.SH","600029.SH","600030.SH",
                 "600036.SH","600048.SH","600050.SH",
                 "600104.SH","600111.SH","600309.SH",
                 "600340.SH","600518.SH","600519.SH",
"600547.SH","600606.SH","600837.SH",
"600887.SH","600919.SH","600958.SH",
"600999.SH","601006.SH","601088.SH",
"601166.SH","601169.SH","601186.SH",
"601211.SH","601229.SH","601288.SH",
"601318.SH", "601328.SH","601336.SH"]
#中国中铁 (601390) 	工商银行 (601398) 	中国太保 (601601)
#中国人寿 (601628) 	中国建筑 (601668) 	中国电建 (601669)
#华泰证券 (601688) 	中国中车 (601766) 	中国交建 (601800)
#光大银行 (601818) 	中国石油 (601857) 	浙商证券 (601878)
#中国银河 (601881) 	中国核电 (601985) 	中国银行 (601988)
#中国重工 (601989) 	洛阳钼业 (603993)

start_date = '2016-01-01'
end_date = '2018-02-07'
fields = ['WIND_CODE','DATE','ADJ_CLOSE','ADJ_OPEN','ADJ_HIGH','ADJ_LOW','VOLUME','PCTCHANGE']


class LSTMModel:
    timesteps = 50
    hidden_layers = 128
    start_date = '2016-01-01'
    end_date = '2018-02-08'
    fields = ['WIND_CODE', 'DATE', 'ADJ_CLOSE', 'ADJ_OPEN', 'ADJ_HIGH', 'ADJ_LOW', 'VOLUME', 'PCTCHANGE']
    global features
    global USECOLS
    global classes
    global model
    global predict_day
    SUB_PATH = ''
    PATH_TO_FILE = ''
    CHECK_POINT_PATH = ''
    BEST_CHECK_POINT_PATH = ''
    # 定义构造方法
    def __init__(self, classes, predict_day):
        self.classes = classes
        self.predict_day=predict_day
        self.SUB_PATH = 'T'+str(self.predict_day)+'/'
        self.model = self.get_model()

    def generate_data(self, wind_code):
        dataset = data_source.getEODData(wind_code, start_date, end_date, fields)
        dataset = dataset.drop(["WIND_CODE", "DATE", "CCY"], 1)
        dataset = np.array(dataset)
        dataset = dataset.tolist()
        X = []
        Y = []
        if(dataset.__len__() - self.timesteps-self.predict_day<0):
            return X, Y
        for i in range(len(dataset) - self.timesteps-self.predict_day+1):
            X.append(dataset[i:i + self.timesteps])
            y = 1
            for j in range(self.predict_day):
                y = dataset[i+self.timesteps+j][4]
            Y.append(y)
        X = np.array(X, dtype=np.float32)
        Y = to_categorical([self.class_generator(y, self.classes) for y in Y], nb_classes=self.classes)
        return X, Y

    def generate_data_from_db(self, wind_code, isNew):
        X = []
        Y = []
        dataset = self.__load_ashareinfo_from_db__(wind_code, isNew)
        if(dataset.__len__()<self.timesteps+ self.predict_day):
            return X,Y
        X.append(dataset[len(dataset) - self.timesteps - self.predict_day:len(dataset)-self.predict_day])
        y=1
        for i in range(self.predict_day):
            y*=(1+dataset[len(dataset)-self.predict_day+i][self.target_col]/100.0)
        y = (y - 1) * 100
        Y.append(y)
        X = np.array(X, dtype=np.float32)
        Y = to_categorical([self.class_generator(y, self.classes) for y in Y], nb_classes=self.classes)
        return X, Y

    def generate_predict_X(self, wind_code):
        X = []
        dataset = Constant.LAST_ASHARE_INFO_DICT[wind_code]
        if(dataset.__len__()<self.timesteps):
            return X
        X.append(dataset[len(dataset) - self.timesteps:len(dataset)])
        X = np.array(X, dtype=np.float32)
        return X

    def class_generator(self, x, c):
        if (c == 2):
            if x < 0: return 0
            return 1
        if (c == 3):
            if x < -0.1: return 0
            if x > 0.1: return 2
            return 1
        if (c == 5):
            if x < -5: return 0
            if x < -1: return 1
            if x < 1: return 2
            if x < 5: return 3
            return 4
        if (c == 7):
            if x < -5: return 0
            if x < -2: return 1
            if x < -0.1: return 2
            if x < 0.1: return 3
            if x < 2: return 4
            if x < 5: return 5
            return 6
        if (c == 9):
            if x < -10: return 0
            if x < -5: return 1
            if x < -2: return 2
            if x < -0.1: return 3
            if x < 0.1: return 4
            if x < 2: return 5
            if x < 5: return 6
            if x < 10: return 7
            return 8


    def __load_ashareinfo_from_db__(self, wind_code, isNew):
        if isNew:
            return Constant.get_AShareInfoHis(wind_code)
        else:
            return Constant.LAST_ASHARE_INFO_DICT[wind_code]

    def get_tensorflow_model(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(10, forget_bias=0.0, state_is_tuple=True)

    def get_model(self):
        net = tflearn.input_data([None, self.timesteps, 6])
        net = tflearn.lstm(net, self.hidden_layers, dropout=0.8,return_seq=True)
        net = tflearn.lstm(net, self.hidden_layers, dropout=0.8, return_seq=True)
        net = tflearn.lstm(net, self.hidden_layers, dropout=0.8)
        net = tflearn.fully_connected(net, self.classes, activation='softmax')
        net = tflearn.regression(net, optimizer='Adam', learning_rate=0.0001, loss='categorical_crossentropy')
        model = tflearn.DNN(net, tensorboard_verbose=0)
        return model


    def splitDataset(self, X, Y, train_validation_ratio):
        train_data_len = int(X.__len__() * train_validation_ratio)
        trainX = X[:train_data_len]
        trainY = Y[:train_data_len]
        validationX = X[train_data_len:]
        validationY = Y[train_data_len:]
        return trainX, trainY, validationX, validationY

    def train_model(self, X, Y, train_validation_ratio, n_epoch, batch_size):
        if(X.__len__()==0):
            print('No Training data. return')
            return
        if (train_validation_ratio == 1):
            self.model.fit(X, Y, show_metric=True,
                           batch_size=batch_size, n_epoch=n_epoch)
        else:
            trainX, trainY, validationX, validationY = self.splitDataset(X, Y, train_validation_ratio)
            self.model.fit(trainX, trainY, validation_set=(validationX, validationY), show_metric=True,
                           batch_size=batch_size, n_epoch=n_epoch)
        return self.model




def train_model(CLASS,predict_day):
    MAX_DATASET_SIZE = 100000
    BATCH_SIZE = 1000
    RATIO = 0.8
    EPOCH = 1
    lstm_model = LSTMModel(CLASS,predict_day)
    X = []
    Y = []
    for i in range(EPOCH):
        for wind_code in wind_code_list:
            global tempX
            global tempY
            tempX, tempY = lstm_model.generate_data(wind_code)
            if (tempY == []):
                print(wind_code + " is new stock and less than 50 Trade Day !")
            else:
                X.extend(tempX)
                Y.extend(tempY)
            print("train model using " + wind_code + " total size=" + Y.__len__().__str__())
            if (Y.__len__() > MAX_DATASET_SIZE):
                lstm_model.train_model(X, Y, RATIO, 1, BATCH_SIZE)
                X.clear()
                Y.clear()
        lstm_model.train_model(X, Y, RATIO, 1, BATCH_SIZE)
        X.clear()
        Y.clear()
        print("Finish EPOCH" + str(i))
    lstm_model.model.save('test2.tfl')

def train_all(class_list,predict_day_list):
    print ('start prediction ....')
    for predict_day in predict_day_list:
        for i in class_list:
            p = Process(target=train_model,args=(i,predict_day))
            p.start()
            p.join()
    print('end prediction ....')

if __name__ == "__main__":
    print ('This is main of module "predict.py"')
    class_list = [3,5]
    predict_day_list = [1,2]
    train_all(class_list,predict_day_list)
    print('Finish main process')
