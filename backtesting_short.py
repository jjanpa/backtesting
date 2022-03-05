import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import datetime
import time
import pandas as pd
import ccxt
import numpy as np
import matplotlib.pyplot as plt

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ########## 변수 설정
        self.flag_mode = False      #데이터 검색 모드 1. 자동  2. 수동
        self.flag_down_data = False     #데이터 다운 모드 False : 다운안함 True : 다운
        #self.trading = 'upbit'
        self.trading = 'binance'
        #self.trader = ccxt.binance
        self.trader = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            }
        })
        self.period = '1m'
        self.start_time = int(pd.Timestamp(2019, 12

                                           , 1, 0, 0, 0).timestamp()) * 1000
        self.delta = 60 * 1000
        self.end_time = int(pd.Timestamp(2021, 6, 14, 9, 0, 0).timestamp()) * 1000
        self.ticker = 'ETH/USDT'
        #self.list_ticker = ['TRX/USDT']
        self.lev = 3
        self.inc = [0.01,0.02,0.025,0.03]
        self.inc_profit = [0.015,0.02,0.025,0.03]
        self.loss_cut = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
        flag_buy = False
        self.plot = True
        self.num_maxdata = 100000
        self.slippage = 0.0025

        self.column_ohlcv = ['timestamp','open','high','low','close','volume']
        self.column_result = ['time','time_buy', 'price_buy', 'time_sell', 'price_sell']
        self.pd_result = pd.DataFrame([(datetime.datetime.now, 0,0,0,0)], columns=self.column_result)
        self.pd_result.drop([0],inplace=True)

        self.column_final_result = ['trader', 'ticker', 'inc', 'inc_profit', 'loss_cut', 'hpr', 'MDD']
        self.temp_result = pd.DataFrame([('binance', 'BTC/KRW', 0,0,0,0,0)], columns=self.column_final_result)
        for year in range(datetime.datetime.fromtimestamp(self.start_time / 1000).year,
                          datetime.datetime.fromtimestamp(self.end_time / 1000).year, 1):
            # print(year)
            str_year = f'{year}'
            self.temp_result[str_year] = 0
        self.temp_result['now'] = 0
        # print(self.temp_result)
        self.temp_result.drop([0], inplace=True)

        ##########
        self.init_API()
        if self.flag_down_data:
            ini_data = self.datadown(self.trader,self.ticker,self.period)
        else:
            ini_data = pd.DataFrame([(0, 0, 0, 0, 0, 0)], columns=self.column_ohlcv)
            ini_data.drop([0], inplace=True)
            num_data = 0
            while True:
                try:
                    temp_path = f'./data/{num_data}_tradername_{self.ticker[0:3]}.xlsx'
                    ini_data = ini_data.append(pd.read_excel(temp_path),ignore_index=True)
                    num_data += 1
                except Exception as e:
                    print('data loading done')
                    break
        ini_data.sort_values(by='timestamp',axis=0,inplace=True)
        print(len(ini_data))
        print(ini_data)
        try:
            for idx_inc in self.inc:
                for idx_profit in self.inc_profit:
                    for idx_loss in self.loss_cut:
                        self.pd_result = pd.DataFrame([(datetime.datetime.now, 0, 0, 0, 0)], columns=self.column_result)
                        self.pd_result.drop([0], inplace=True)
                        pre_hpr = 1
                        data = ini_data.copy()
                        for i in range(0,len(data),1):
                            now_data = data.iloc[i]
                            print(data.iloc[i]['time'])
                            try:
                                if flag_buy == False:
                                    if (pre_data['close'] - pre_data['open'])/pre_data['open'] > idx_inc:
                                        time_buy = now_data['timestamp']
                                        price_buy = now_data['open']
                                        price_target = price_buy * (1+idx_profit)
                                        price_loss = price_buy * (1-idx_loss)
                                        flag_buy = True
                                if flag_buy:
                                    if now_data['high'] >price_target:
                                        time_sell = now_data['timestamp']
                                        price_sell = price_target
                                        temp = pd.DataFrame([(datetime.datetime.fromtimestamp(time_buy/1000), time_buy,price_buy,time_sell,price_sell)], columns=self.column_result)
                                        #print(temp)
                                        self.pd_result = self.pd_result.append(temp,True)
                                        flag_buy = False
                                    elif now_data['low'] < price_loss:
                                        time_sell = now_data['timestamp']
                                        price_sell = price_loss
                                        temp = pd.DataFrame([(datetime.datetime.fromtimestamp(time_buy / 1000), time_buy, price_buy,
                                                              time_sell, price_sell)], columns=self.column_result)
                                        #print(temp)
                                        self.pd_result = self.pd_result.append(temp, True)
                                        flag_buy = False
                                pre_data = now_data
                            except Exception as e:
                                pre_data = now_data
                                print(e)
                        self.pd_result['ror'] = self.lev * (self.pd_result['price_sell']-self.pd_result['price_buy'])/self.pd_result['price_buy'] + 1- self.slippage
                        self.pd_result['hpr'] = self.pd_result['ror'].cumprod()
                        self.pd_result['dd'] = (self.pd_result['hpr'].cummax() - self.pd_result['hpr']) / self.pd_result['hpr'].cummax() * 100
                        #print(self.pd_result)
                        print("inc :",idx_inc)
                        print('profit : ',idx_profit)
                        print('loss cut : ',idx_loss)
                        print("hrp : ", self.pd_result.iloc[-1]['hpr'])
                        print("MDD(%): ", self.pd_result['dd'].max())
                        temp_path = f'./result/detail/d_result_{idx_inc}_{idx_profit}_{idx_loss}.xlsx'
                        self.pd_result.to_excel(temp_path)
                        result = pd.DataFrame(
                            [(self.trader, self.ticker, idx_inc, idx_profit, idx_loss, self.pd_result.iloc[-1]['hpr'], self.pd_result['dd'].max())],
                            columns=self.column_final_result)
                        pre_hpr = 1
                        for year in range(datetime.datetime.fromtimestamp(self.start_time / 1000).year,
                                          datetime.datetime.fromtimestamp(self.end_time / 1000).year, 1):
                            print(year)
                            str_year = f'{year}'
                            for i in range(0, len(self.pd_result), 1):
                                if self.pd_result.iloc[i]['time'].year == year + 1:
                                    result.loc[0, str_year] = self.pd_result.iloc[i-1]['hpr'] / pre_hpr
                                    pre_hpr = self.pd_result.iloc[i-1]['hpr']
                                    print(self.pd_result.iloc[i-1]['time'])
                                    break
                        result.loc[0, 'now'] = self.pd_result.iloc[-1]['hpr'] / pre_hpr
                        self.temp_result = self.temp_result.append(result, ignore_index=True)
                        if self.plot:
                            plt.figure(figsize=(10, 10))
                            plt.plot(self.pd_result['time'], self.pd_result['hpr'], 'b-', label='yield')
                            plt.plot(data['time'], data['ratio'], 'r-', label=self.ticker)
                            title = self.ticker
                            plt.title(title)
                            plt.legend(loc='best')
                            plt.grid(True)
                            #plt.show()
                            path_fig = f'./graph/graph_{idx_inc}_{idx_profit}_{idx_loss}.png'
                            plt.savefig(path_fig)
                    temp_path = f'./result/result_{idx_inc}_{idx_profit}.xlsx'
                    self.temp_result.to_excel(temp_path)
                    self.temp_result = pd.DataFrame([('binance', 'BTC/KRW', 0, 0, 0, 0, 0)],
                                                    columns=self.column_final_result)
                    for year in range(datetime.datetime.fromtimestamp(self.start_time / 1000).year,
                                      datetime.datetime.fromtimestamp(self.end_time / 1000).year, 1):
                        # print(year)
                        str_year = f'{year}'
                        self.temp_result[str_year] = 0
                    self.temp_result['now'] = 0
                    self.temp_result.drop([0], inplace=True)
        except Exception as e:
            print(e)
        print(self.temp_result)

    def init_API(self):
        if self.trading == 'binance':
            self.trader = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'
                }
            })
        elif self.trading == 'upbit':
            self.trader = ccxt.upbit()
        print("로그인")

    def datadown(self,tradername, ticker, period):
        try:
            temp_ohlcv = pd.DataFrame([(0, 0.0, 0.0, 0.0, 0.0, 0.0)],columns=self.column_ohlcv)
            temp_ohlcv.drop([0],inplace=True)
            print(f'data down {ticker} {period}')
            if period == '1m':
                delta = 60*1000
            elif period == '15m':
                delta = 15*60*1000
            elif period == '1d':
                delta = 24*60*60*1000
            elif period == '3d':
                delta = 3 * 24 * 60 * 60 * 1000
            elif period == '1h':
                delta = 60*60*1000
            elif period == '12h':
                delta = 12*60*60*1000
            n_time = self.start_time
            num_data = 0
            try:
                while self.end_time >= n_time:
                    print(...)
                    time.sleep(0.15)
                    ohlcv = self.trader.fetchOHLCV(ticker, period, n_time)
                    ohlcv = pd.DataFrame(ohlcv,columns=self.column_ohlcv)
                    if len(ohlcv)!=0:
                        temp_ohlcv = temp_ohlcv.append(ohlcv,ignore_index=True)
                        n_time = int(ohlcv.iloc[-1]['timestamp'])
                        print(f'data down {datetime.datetime.fromtimestamp(n_time/1000)}')
                        n_time = n_time + delta
                    else:
                        n_time = n_time + delta
            except Exception as e:
                print('err',e)
            #temp_ohlcv.to_excel(path)
            start_price = temp_ohlcv.loc[0,'close']
            list_time = []
            list_ratio = []
            len_data = len(temp_ohlcv)
            for i in range(0,len_data,1):
                print(len_data,' ', i)
                list_time.append(datetime.datetime.fromtimestamp(temp_ohlcv.loc[i,'timestamp'] / 1000))
                list_ratio.append(temp_ohlcv.loc[i,'close']/start_price)
            temp_ohlcv['time'] = list_time
            temp_ohlcv['ratio'] = list_ratio
            print('make data')
            temp_save = temp_ohlcv.copy()
            print('copy data')

            while True:
                print(len(temp_save))
                if len(temp_save) > self.num_maxdata:
                    savedata = temp_save[0:self.num_maxdata]
                    temp_path = f'./data/{num_data}_tradername_{ticker[0:3]}.xlsx'
                    print(temp_path)
                    num_data = num_data + 1
                    savedata.to_excel(temp_path)
                    temp_save = temp_save[self.num_maxdata:].copy()
                else:
                    temp_path = f'./data/{num_data}_tradername_{ticker[0:3]}.xlsx'
                    temp_save.to_excel(temp_path)
                    break
            #temp_ohlcv = temp_ohlcv.set_index('timestamp')
            #temp_ohlcv.to_excel(temp_path)
            print(f'data down done')
            print(temp_ohlcv)
            return temp_ohlcv
        except Exception as e:
            print(f'data down fail : {e}')

app = QApplication(sys.argv)
window = MyWindow()
#window.show()
app.exec_()