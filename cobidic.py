#### 마지막 수정 23.11.25

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
        self.flag_down_data = True     #데이터 다운 모드 False : 다운안함 True : 다운

        self.plot = True        #그래프 표시여부
        self.detail_save = True     #디테일 데이터 저장여부

        self.slippage = 0.002
        self.gap = 0.0
        self.factor = 1

        self.trader = 'binance'
        ticker_list = ['ETH']
        self.ticker = {f'KRW-{key}': 1/len(ticker_list) for key in dict.fromkeys(ticker_list).keys()}     #ticker list

        if self.trader == 'binance_m':
            self.factor = 3

        ### DataFrame 선언
        self.column_ohlcv = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.column_data = ['timestamp','open', 'high', 'low', 'close', 'volume','time','ratio']
        self.column_result = ['trader', 'ticker', 'hpr', 'MDD']

        self.start_time = datetime.datetime(2023, 1, 1, 9)
        self.end_time = datetime.datetime(2023,12,11,20)
        self.temp_result = pd.DataFrame([('binance','BTC/KRW',0.0,0.0)], columns=self.column_result)

        # minute1/minute3/minute5/minute10/minute15/minute30/minute60/minute240/day/week/month
        self.period = 'minute5'  #데이터 다운로드 period

        # delta 설정
        # minute1/minute3/minute5/minute10/minute15/minute30/minute60/minute240/day/week/month
        if self.period == 'minute1':
            delta = datetime.timedelta(minutes=1)
        elif self.period == 'minute3':
            delta = datetime.timedelta(minutes=3)
        elif self.period == 'minute5':
            delta = datetime.timedelta(minutes=5)
        elif self.period == 'minute10':
            delta = datetime.timedelta(minutes=10)
        elif self.period == 'minute15':
            delta = datetime.timedelta(minutes=15)
        elif self.period == 'minute30':
            delta = datetime.timedelta(minutes=30)
        elif self.period == 'minute60':
            delta = datetime.timedelta(minutes=60)
        elif self.period == 'minute240':
            delta = datetime.timedelta(minutes=240)
        elif self.period == 'day':
            delta = datetime.timedelta(days=1)
        elif self.period == 'week':
            delta = datetime.timedelta(weeks=1)

        # self.trader = pyupbit

        self.save_file = f'./result/result_cobidic_{self.period}_{self.end_time.year}_{self.end_time.month}_{self.end_time.day}.csv'
        try:
            self.result_file = pd.read_csv(self.save_file)
            self.result_file.set_index(self.result_file.columns[0],inplace=True)
            self.temp_result = self.result_file
        except Exception as e:
            print(e.__traceback__)
            self.temp_result = pd.DataFrame([('binance','BTC/KRW',0.0,0.0)], columns=self.column_result)

            for year in range(self.start_time.year,self.end_time.year,1):
                str_year = f'{year}'
                self.temp_result[str_year] = 0
            self.temp_result['now'] = 0
            self.temp_result.drop([0], inplace=True)
            self.temp_result.to_csv(self.save_file)

        for ticker in self.ticker.keys():
            # ticker별 ohlcv 데이터 다운로드
            base = ticker.split('-')[1]
            quote = ticker.split('-')[0]
            path = f'./data/{self.period}_{base}_{quote}_{self.start_time.timestamp()}_{self.end_time.timestamp()}.csv'

            data = func.datadown(self.trader, self.start_time, self.end_time, ticker, self.period, True, path)
            data.reset_index(inplace=True)

            data['sto_K'], data['sto_D'], data['sto_J'] = compute_stochastic_oscillator(data)

            data = RSI(data,14)

            data['ema_short'] = data['close'].ewm(12).mean() #12번간 지수 이동평균
            data['ema_long'] = data['close'].ewm(26).mean() #26번간 지수 이동평균

            data['macd'] = data['ema_short'] - data['ema_long']

            # print(data)

            data.to_csv(f'./result/result_sum.csv')

            flag_sto = False
            flag_RSI = False
            buy_price = 0
            hpr = 1

            for i in range(len(data)):
                tt = data.iloc[i]
                if flag_sto == False and tt['sto_K'] < 20 and tt['sto_D'] < 20:
                    flag_sto = True
                    # print('get sto!!',tt['time'],tt['sto_K'])

                if tt['sto_K'] > 80 or tt['sto_D'] > 80:
                    # if flag_sto == True:
                        # print('lose sto!!',tt['time'],tt['sto_K'])
                    flag_sto = False

                if flag_sto == True:
                    if tt['RSI'] > 50:
                        # if flag_RSI == False:
                            # print('get RSI!!', tt['time'],tt['RSI'])
                        flag_RSI = True
                    else:
                        flag_RSI = False
                else:
                    flag_RSI = False

                if flag_sto == True and flag_RSI == True:
                    if tt['ema_short'] > tt['ema_long']:
                        if buy_price == 0:
                            buy_price = tt['close']
                            # print('buy!!', tt['time'], tt['close'])
                            num = i
                            while num > 2:
                                # print(num)
                                if data.iloc[num-1]['low'] > data.iloc[num]['low']:break
                                num-=1
                            loss_cut = data.iloc[num]['low']
                            profit = 1.5*(buy_price - loss_cut) + buy_price

                            print('loss cut!!! : ',loss_cut)
                            print('stop cut!!! : ',profit)


                if buy_price != 0:
                    if tt['close'] < loss_cut or tt['close'] > profit:
                        ror = tt['close'] / buy_price
                        hpr *= ror
                        print(ror,' sell : ',tt['close'],'buy : ',buy_price,'hpr : ', hpr)
                        buy_price = 0

        # 그래프 저장
        if self.plot:
            plt.figure(figsize=(10, 10))
            fig, ax1 = plt.subplots()

            ax1.plot(data['time'], data['ratio'], 'r-', label=ticker)
            ax2 = ax1.twinx()
            ax2.plot(data['time'], data['RSI'], 'b-', label='yield')
            ax3 = ax1.twinx()
            ax3.plot(data['time'], data['macd'], 'g-', label='yield')

            title = ticker
            plt.title(title)
            plt.legend(loc='best')
            plt.grid(True)
            base = ticker.split('-')[1]
            quote = ticker.split('-')[0]
            temp_path = f'./graph/graph_{base}_{quote}.png'
            plt.savefig(temp_path)
            plt.close('all')
            plt.clf()
            # plt.show()

def RSI(df,interval):
    df['U'] = np.where(df.diff(1)['close'] > 0, df.diff(1)['close'], 0)
    df['D'] = np.where(df.diff(1)['close'] < 0, df.diff(1)['close']*(-1), 0)
    df['AU'] = df['U'].rolling(interval).mean()
    df['AD'] = df['D'].rolling(interval).mean()
    df['RSI'] = (df['AU'] / (df['AU']+df['AD']))*100
    df['RSI_Signal']= df['RSI'].rolling(9).mean()
    del df['U'], df['D'], df['AU'], df['AD']

    return df

def compute_stochastic_oscillator(df, n=14, m=3, t=3):
    low_min = df['low'].rolling(window=n).min()
    high_max = df['high'].rolling(window=n).max()
    K = (df['close'] - low_min) / (high_max - low_min) * 100
    D = K.rolling(window=m).mean()
    J = D.rolling(window=t).mean()
    return K, D, J



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')
    main()
