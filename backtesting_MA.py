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

        self.plot = False        #그래프 표시여부
        self.detail_save = True     #디테일 데이터 저장여부

        self.slippage = 0.002
        self.gap = 0.0

        # self.numforma1 = [x for x in range(1,200,1)]
        self.numforma1 = [14]
        # self.numforma2 = [x for x in range(1,200,1)]
        self.numforma2 = [70]
        # self.numforma3 = [x for x in range(1,100,1)]
        self.numforma3 = [10]
        # self.numforma4 = [x for x in range(1,100,1)]
        self.numforma4 = [55]

        self.ma_flow = [370]
        # self.ma_flow = [x for x in range(1,600,1)]

        ticker_list = ['ETH']
        self.ticker = {f'KRW-{key}': 1/len(ticker_list) for key in dict.fromkeys(ticker_list).keys()}     #ticker list

        ### DataFrame 선언
        self.column_ohlcv = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.column_data = ['timestamp','open', 'high', 'low', 'close', 'volume','time','ratio']
        self.column_result = ['trader', 'ticker', 'ma1', 'ma2', 'ma3', 'ma4','ma_set', 'hpr', 'MDD']

        self.start_time = datetime.datetime(2016, 1, 1, 9)
        self.end_time = datetime.datetime(2022, 8, 7, 9)
        self.temp_result = pd.DataFrame([('binance','BTC/KRW',0.0,0.0,0.0,0.0,0.0,0.0,0.0)], columns=self.column_result)

        # minute1/minute3/minute5/minute10/minute15/minute30/minute60/minute240/day/week/month
        self.period = 'minute60'  #데이터 다운로드 period

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

        # self.trader = ccxt.upbit()
        self.trader = pyupbit

        self.save_file = f'./result/result_{self.period}_{self.end_time.year}_{self.end_time.month}_{self.end_time.day}.csv'
        try:
            self.result_file = pd.read_csv(self.save_file)
            self.result_file.set_index(self.result_file.columns[0],inplace=True)
            self.temp_result = self.result_file
            saved_ma1 = int(self.result_file.tail(1)['ma1'])
            saved_ma2 = int(self.result_file.tail(1)['ma2'])
            saved_ma_set = int(self.result_file.tail(1)['ma_set'])
        except Exception as e:
            print(e.__traceback__)
            self.temp_result = pd.DataFrame([('binance','BTC/KRW',0.0,0.0,0.0,0.0,0.0,0.0,0.0)], columns=self.column_result)

            for year in range(self.start_time.year,self.end_time.year,1):
                #print(year)
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

            if self.flag_down_data:  # 데이터를 직접 다운로드
                data = func.datadown(self.trader, self.start_time - max(self.numforma4 + self.ma_flow) * delta, self.end_time, ticker, self.period, True, path)
            else:  # 기존 저장된 데이터를 불러옴
                try:
                    data = pd.read_csv('./data.csv')
                    # data.set_index(keys='timestamp', inplace=True)
                    print(data)
                    list_timestamp = []
                    for i in range(len(data)):
                        # print(data.iloc[i]['time'])
                        tt = data.iloc[i]['time']
                        # print(tt.split('-')[0])
                        # print(datetime.datetime(year=int(tt.split('-')[0]),month=int(tt.split('-')[1]),day=int(tt.split('-')[2])))
                        tt = datetime.datetime(year=int(tt.split('-')[0]),month=int(tt.split('-')[1]),day=int(tt.split('-')[2]))
                        print(tt)
                        list_timestamp.append(datetime.datetime.timestamp(tt))

                    data['timestamp'] = list_timestamp
                    data['ratio'] = 1

                    print(data)
                except Exception as e:
                    print(e)
                    print('no saved data')
                    data = func.datadown(self.trader, self.start_time, self.end_time, ticker, self.period, True, path)
            data.reset_index(inplace=True)
            flag_new = False

            for ma1 in self.numforma1:
                for ma2 in self.numforma2:
                    # for ma3 in self.numforma3:
                    #     for ma4 in self.numforma4:
                    for ma_set in self.ma_flow:
                        # if ma2 > ma1 and ma4 > ma3:
                        ma3 = ma1
                        ma4 = ma2
                        flag_new = False
                        print(f'moving ave_{ma1}/{ma2}')
                        print(f'moving ave_{ma3}/{ma4}')
                        print(f'set ma_{ma_set}')
                        if ma2 > ma1 and ma4 > ma3:
                            try:
                                if ma1 < saved_ma1:
                                    continue
                                if ma2 < saved_ma2:
                                    continue
                                if ma_set <= saved_ma_set:
                                    continue
                            except Exception as e:
                                print(e)

                            flag_new = True

                            pre_hpr = 1

                            data_copy = data.copy(False)
                            data_copy[ma1] = data_copy['close'].rolling(ma1).mean().shift(1)    # 이전날 세팅된 값 shift로 끌어옴
                            data_copy[ma2] = data_copy['close'].rolling(ma2).mean().shift(1)    # 이전날 세팅된 값 shift로 끌어옴
                            data_copy[ma3] = data_copy['close'].rolling(ma3).mean().shift(1)    # 이전날 세팅된 값 shift로 끌어옴
                            data_copy[ma4] = data_copy['close'].rolling(ma4).mean().shift(1)    # 이전날 세팅된 값 shift로 끌어옴

                            data_copy['ma_set'] = data_copy['close'].rolling(ma_set).mean().shift(1)    # 상승장 하락장 판별할 기준 MA

                            data_copy = data_copy[max(ma1, ma2, ma3, ma4, max(self.ma_flow)):].reset_index(drop=True)
                            data_copy['ratio'] = data_copy['ratio'] / data_copy['ratio'][0]

                            strat_year = datetime.datetime.fromtimestamp(int(data_copy['timestamp'][1])).year

                            #
                            # flag_sys = 1
                            # time_buy = 0
                            # time_sell = 0
                            # price_buy = 0
                            # price_sell = 0
                            #
                            # hpr = 1
                            #
                            # list_time = []
                            # list_time_buy = []
                            # list_time_sell = []
                            # list_hpr = []
                            # list_ror = []
                            # list_buy = []
                            # list_sell = []
                            # list_ratio = []
                            #
                            # start = datetime.datetime.fromtimestamp(data_copy['timestamp'][0])
                            #
                            # save_time = datetime.datetime(year=start.year, month=start.month, day=start.day, hour=9)
                            # if start > save_time:
                            #     save_time = save_time + datetime.timedelta(days=1)
                            #
                            # for ind in range(1,len(data_copy),1):
                            #     now = datetime.datetime.fromtimestamp(data_copy['timestamp'][ind])
                            #     now_data = data_copy.iloc[ind]
                            #     pre_data = data_copy.iloc[ind-1]
                            #
                            #
                            #     if now >= save_time:
                            #         save_time = save_time + datetime.timedelta(days=1)
                            #         # list_time.append(now)
                            #         # list_time_buy.append(time_buy)
                            #         # list_time_sell.append(time_sell)
                            #         # list_buy.append(price_buy)
                            #         # list_sell.append(price_sell)
                            #         # list_hpr.append(hpr)
                            #         # list_ratio.append(now_data['ratio'])
                            #         # if not price_sell == 0:
                            #         #     time_buy = 0
                            #         #     time_sell = 0
                            #         #     price_buy = 0
                            #         #     price_sell = 0
                            #
                            #     if flag_sys == 1:
                            #         try:
                            #             flag_set = False
                            #             # for k in range(1,3,1):
                            #             #     if list_ror[-1 * k] < 0:
                            #             #         flag_set = False
                            #             if flag_set:
                            #                 if now_data[ma1] > now_data[ma2] and pre_data[ma1] < pre_data[ma2]:
                            #                     price_buy = now_data['open']
                            #                     # print(now,' : buy : ',price_buy)
                            #                     time_buy = now
                            #                     flag_sys = 2
                            #             else:
                            #                 if now_data[ma1] > now_data[ma2] and pre_data[ma1] < pre_data[ma2] and data_copy.iloc[ind]['ma_set'] > data_copy.iloc[ind-1]['ma_set']:
                            #                     price_buy = now_data['open']
                            #                     # print(now,' : buy : ',price_buy)
                            #                     time_buy = now
                            #                     flag_sys = 2
                            #         except Exception as e:
                            #             print(e)
                            #             if now_data[ma1] > now_data[ma2] and pre_data[ma1] < pre_data[ma2]:
                            #                 price_buy = now_data['open']
                            #                 # print(now,' : buy : ',price_buy)
                            #                 time_buy = now
                            #                 flag_sys = 2
                            #     elif flag_sys == 2:
                            #         if now_data[ma3] < now_data[ma4] and pre_data[ma3] > pre_data[ma4]:
                            #             price_sell = now_data['open']
                            #             time_sell = now
                            #             hpr = hpr * (price_sell / price_buy - self.slippage)
                            #             ror = price_sell / price_buy - 1 - self.slippage
                            #             list_time_buy.append(time_buy)
                            #             list_time_sell.append(time_sell)
                            #             list_buy.append(price_buy)
                            #             list_sell.append(price_sell)
                            #             list_hpr.append(hpr)
                            #             list_ror.append(ror)
                            #             # print(f'{now}  sell : {price_sell},  {ror*100:,.2f}  {hpr*100:,.2f}')
                            #             flag_sys = 1
                            #
                            # temp_data = pd.DataFrame()
                            # temp_data['time buy'] = list_time_buy
                            # temp_data['time sell'] = list_time_sell
                            # temp_data['price buy'] = list_buy
                            # temp_data['price sell'] = list_sell
                            # temp_data['hpr'] = list_hpr
                            # temp_data['dd'] = (temp_data['hpr'].cummax() - temp_data['hpr']) / temp_data['hpr'].cummax() * 100
                            #
                            # print("ticker : ", ticker)
                            # print(f'MA : {ma1}/{ma2}')
                            # print(f'MA : {ma3}/{ma4}')
                            # print("hrp : ", temp_data.iloc[-1]['hpr'])
                            # print("MDD(%): ", temp_data['dd'].max(), "\n")

                            data_copy['increase'] = np.where(data_copy[ma1] > data_copy[ma2], np.where(
                                data_copy[ma2].shift(1) > data_copy[ma1].shift(1),
                                np.where(data_copy['ma_set'] > data_copy['ma_set'].shift(1), 1, 0), 0), 0)
                            # data_copy['increase'] = np.where(data_copy[ma1] > data_copy[ma2], np.where(
                            #     data_copy[ma2].shift(1) > data_copy[ma1].shift(1),1, 0), 0)
                            # data_copy['increase_shift'] = data_copy['increase'].shift(1)

                            data_copy['price buy'] = np.where(data_copy['increase'] == 1, data_copy['open'], 0)

                            data_copy['price sell'] = np.where(data_copy[ma2] > data_copy[ma1], 1, 0)
                            data_copy['price sell'] = np.where(data_copy['price sell'] == 1,
                                                               np.where(data_copy['price sell'].shift(1) == 0,
                                                                        data_copy['open'], 0), 0)
                            is_buy = data_copy['price buy'] != 0
                            is_sell = data_copy['price sell'] != 0

                            data_copy = data_copy[is_buy | is_sell]

                            data_copy['price sell'] = np.where(data_copy['price sell'] != 0,
                                                               np.where(data_copy['price buy'].shift(1) != 0,
                                                                        data_copy['price sell'], 0), 0)
                            data_copy['price buy'] = np.where(data_copy['price buy'].shift(1) != 0, 0,
                                                              data_copy['price buy'])

                            is_buy = data_copy['price buy'] != 0
                            is_sell = data_copy['price sell'] != 0

                            data_copy = data_copy[is_buy | is_sell]

                            data_copy['ror'] = np.where(data_copy['price sell'] != 0,
                                                        data_copy['price sell'] / data_copy['price buy'].shift(
                                                            1) - self.slippage, 1)

                            data_copy['time buy'] = data_copy['time'].shift(1)

                            data_copy['time sell'] = data_copy['time']

                            data_copy['price buy'] = data_copy['price buy'].shift(1)

                            data_copy = data_copy[['time buy', 'time sell', 'price buy', 'price sell', 'ror']]

                            is_ror = data_copy['price sell'] != 0

                            data_copy = data_copy[is_ror]

                            data_copy['hpr'] = data_copy['ror'].cumprod()
                            data_copy['dd'] = (data_copy['hpr'].cummax() - data_copy['hpr']) / data_copy[
                                'hpr'].cummax() * 100

                            temp_data = data_copy

                            print("ticker : ", ticker)
                            print(f'MA : {ma1}/{ma2}')
                            print(f'MA : {ma3}/{ma4}')
                            print(f'MA set : {ma_set}')
                            print("hrp : ", temp_data.iloc[-1]['hpr'])
                            print("MDD(%): ", temp_data['dd'].max(), "\n")

                            # ticker별 결과 저장
                            base = ticker.split('-')[1]
                            quote = ticker.split('-')[0]
                            temp_path = f'./result/detail/d_result_{base}_{quote}_{ma1}_{ma2}_{ma3}_{ma4}_{ma_set}.csv'
                            if self.detail_save:
                                temp_data.to_csv(temp_path)

                            result = pd.DataFrame([('upbit', ticker, ma1, ma2, ma3, ma4, ma_set, temp_data.iloc[-1]['hpr'], temp_data['dd'].max())],
                                                  columns=self.column_result)

                            # 년도별 수익률 계산 및 저장
                            for year in range(strat_year,
                                              self.end_time.year, 1):
                                str_year = f'{year}'
                                for i in range(len(temp_data)):
                                    try:
                                        time_year = temp_data.iloc[i]['time sell'].year
                                    except:
                                        time_year = datetime.datetime.fromtimestamp(temp_data.index[i]/1000).year
                                    if time_year == year + 1:
                                        result.loc[0, str_year] = temp_data.iloc[i-1]['hpr'] / pre_hpr
                                        pre_hpr = temp_data.iloc[i-1]['hpr']
                                        break
                            result.loc[0, 'now'] = temp_data.iloc[-1]['hpr'] / pre_hpr
                            self.temp_result = pd.concat([self.temp_result, result])

                            # 그래프 저장
                            if self.plot:
                                plt.figure(figsize=(10, 10))
                                fig, ax1 = plt.subplots()

                                ax1.plot(data_copy['time'], data_copy['ratio'], 'r-', label=ticker)
                                ax2 = ax1.twinx()
                                ax2.plot(temp_data['time sell'], temp_data['hpr'], 'b-', label='yield')

                                title = ticker
                                plt.title(title)
                                plt.legend(loc='best')
                                plt.grid(True)
                                base = ticker.split('-')[1]
                                quote = ticker.split('-')[0]
                                temp_path = f'./graph/graph_{base}_{quote}_{ma1}_{ma2}_{ma3}_{ma4}.png'
                                plt.savefig(temp_path)
                                plt.close('all')
                                plt.clf()
                                # plt.show()
                    if flag_new:
                        self.temp_result.to_csv(f'./result/result_{self.period}_{self.end_time.year}_{self.end_time.month}_{self.end_time.day}_sum.csv')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')
    main()
