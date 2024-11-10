import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyupbit

import func_backtesting as func


class main():
    def __init__(self):
        super().__init__()

        tmp = pyupbit.get_ohlcv('KRW-ETH', interval="minute60", count=500)
        print(tmp)
        print('end')

        self.slippage = 0.002

        # self.numforma1 = [x for x in range(1,20,1)]
        self.numforma1 = [10]
        # self.numforma2 = [x for x in range(10,50,1)]
        self.numforma2 = [55]
        # self.numforma3 = [x for x in range(1,100,1)]
        self.numforma3 = [494]

        data = pd.read_csv("./data.csv")


        data_copy = data.copy()

        ma1 = 10
        ma2 = 55
        ma3 = 494


        data_copy['ma1'] = data_copy['close'].rolling(ma1).mean().shift(1)  # 이전날 세팅된 값 shift로 끌어옴
        data_copy['ma2'] = data_copy['close'].rolling(ma2).mean().shift(1)  # 이전날 세팅된 값 shift로 끌어옴

        data_copy['ma3'] = data_copy['close'].rolling(ma3).mean().shift(1)  # 이전날 세팅된 값 shift로 끌어옴

        data_copy = data_copy[ma3:].reset_index(drop=True)

        data_copy['increase'] = np.where(data_copy['ma1'] > data_copy['ma2'], np.where(data_copy['ma2'].shift(1) > data_copy['ma1'].shift(1), np.where(data_copy['ma3'] > data_copy['ma3'].shift(1), 1, 0), 0), 0)
        # data_copy['increase_shift'] = data_copy['increase'].shift(1)

        data_copy['price_buy'] = np.where(data_copy['increase'] == 1, data_copy['open'], 0)

        data_copy['price_sell'] = np.where(data_copy['ma2'] > data_copy['ma1'], 1, 0)
        data_copy['price_sell'] = np.where(data_copy['price_sell'] == 1, np.where(data_copy['price_sell'].shift(1) == 0, data_copy['open'], 0), 0)

        is_buy = data_copy['price_buy'] != 0
        is_sell = data_copy['price_sell'] != 0

        data_copy = data_copy[is_buy | is_sell]

        data_copy['price_sell'] = np.where(data_copy['price_sell'] != 0, np.where(data_copy['price_buy'].shift(1) != 0, data_copy['price_sell'], 0), 0)
        data_copy['price_buy'] = np.where(data_copy['price_buy'].shift(1) != 0, 0, data_copy['price_buy'])

        is_buy = data_copy['price_buy'] != 0
        is_sell = data_copy['price_sell'] != 0

        data_copy = data_copy[is_buy | is_sell]

        data_copy['ror'] = np.where(data_copy['price_sell'] != 0, data_copy['price_sell'] / data_copy['price_buy'].shift(1) - self.slippage, 1)

        data_copy['time_buy'] = np.where(data_copy['price_sell'] != 0, data_copy['time'].shift(1), 0)

        data_copy['time_sell'] = data_copy['time']

        data_copy['price_buy'] = data_copy['price_buy'].shift(1)

        data_copy = data_copy[['time_buy', 'time_sell', 'price_buy', 'price_sell', 'ror']]

        is_ror = data_copy['price_sell'] != 0

        data_copy = data_copy[is_ror]

        data_copy['hpr'] = data_copy['ror'].cumprod()
        data_copy['dd'] = (data_copy['hpr'].cummax() - data_copy['hpr']) / data_copy['hpr'].cummax() * 100

        print("hrp : ", data_copy.iloc[-1]['hpr'])
        print("MDD(%): ", data_copy['dd'].max(), "\n")

        data_copy.to_csv('./data_result.csv')




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')
    main()
