from test_data import LSTMModel
from predictor.models import PredictResult
import os
import time
import numpy as np
from multiprocessing import Process


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


def predict(CLASS,predict_day):
    lstm_model = LSTMModel(CLASS,predict_day)
    lstm_model.model.load('test.tfl')
    date=time.strftime("%Y%m%d", time.localtime())
    for wind_code in wind_code_list:
        X = wind_code_list
        if X == []:
            print(wind_code + " is new stock and less than 50 Trade Day !")
        else:
            result = lstm_model.model.predict(X)
            if (CLASS == 2):
                if(predict_day==1):
                    res = PredictResult.objects.get(S_INFO_WINDCODE=wind_code)
                    res.T1_C2_PROB = result[0][0]
                if(predict_day ==2):
                    res = PredictResult.objects.get(S_INFO_WINDCODE=wind_code)
                    res.T2_C2_PROB = result[0][0]
                if(predict_day ==3):
                    res = PredictResult.objects.get(S_INFO_WINDCODE=wind_code)
                    res.T3_C2_PROB = result[0][0]
                if(predict_day ==4):
                    res = PredictResult.objects.get(S_INFO_WINDCODE=wind_code)
                    res.T4_C2_PROB = result[0][0]
                if(predict_day ==5):
                    res = PredictResult.objects.get(S_INFO_WINDCODE=wind_code)
                    res.T5_C2_PROB = result[0][0]
                if(predict_day ==10):
                    res = PredictResult(S_INFO_WINDCODE=wind_code,
                                    PREDICT_DT=date,
                                    T10_C2_PROB=result[0][0])
            if (CLASS == 3):
                C3_CLASS = np.array(result[0]).argmax()
                C3_PROB = max(result[0])
                res = PredictResult.objects.get(S_INFO_WINDCODE=wind_code)
                if (predict_day == 1):
                    res.T1_C3_CLASS = C3_CLASS
                    res.T1_C3_PROB = C3_PROB
                if (predict_day == 2):
                    res.T2_C3_CLASS = C3_CLASS
                    res.T2_C3_PROB = C3_PROB
                if (predict_day == 3):
                    res.T3_C3_CLASS = C3_CLASS
                    res.T3_C3_PROB = C3_PROB
                if (predict_day == 4):
                    res.T4_C3_CLASS = C3_CLASS
                    res.T4_C3_PROB = C3_PROB
                if (predict_day == 5):
                    res.T5_C3_CLASS = C3_CLASS
                    res.T5_C3_PROB = C3_PROB
                if (predict_day == 10):
                    res.T10_C3_CLASS = C3_CLASS
                    res.T10_C3_PROB = C3_PROB
            if (CLASS == 7):
                C7_CLASS = np.array(result[0]).argmax()
                C7_PROB = max(result[0])
                res = PredictResult.objects.get(S_INFO_WINDCODE=wind_code)
                if (predict_day == 1):
                    res.T1_C7_CLASS = C7_CLASS
                    res.T1_C7_PROB = C7_PROB
                if (predict_day == 2):
                    res.T2_C7_CLASS = C7_CLASS
                    res.T2_C7_PROB = C7_PROB
                if (predict_day == 3):
                    res.T3_C7_CLASS = C7_CLASS
                    res.T3_C7_PROB = C7_PROB
                if (predict_day == 4):
                    res.T4_C7_CLASS = C7_CLASS
                    res.T4_C7_PROB = C7_PROB
                if (predict_day == 5):
                    res.T5_C7_CLASS = C7_CLASS
                    res.T5_C7_PROB = C7_PROB
                if (predict_day == 10):
                    res.T10_C7_CLASS = C7_CLASS
                    res.T10_C7_PROB = C7_PROB
            res.save()

def predict_all(class_list,predict_day_list):
    print ('start prediction ....')
    for predict_day in predict_day_list:
        for i in class_list:
            print('start prediction: class'+ str(i)+' predict day:'+str(predict_day))
            p = Process(target=predict,args=(i,predict_day))
            p.start()
            p.join()
    print('end prediction ....')

if __name__ == "__main__":
    print ('This is main of module "predict.py"')
    class_list = [2, 3, 7]
    predict_day_list=[10,5,4,3,2,1]
    predict_all(class_list,predict_day_list)
    print('Finish main process')
