import numpy as np
import sys
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

        ### DataFrame 선언
        self.column_ohlcv = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.column_data = ['timestamp','open', 'high', 'low', 'close', 'volume','time','ratio']
        self.column_result = ['trader', 'ticker', 'k','var','leverage','hpr','MDD']
        self.column_result_falling = ['trader', 'ticker', 'k','var','leverage','losscut','hpr','MDD']

        ########## 변수 설정
        self.flag_mode = False   #데이터 검색 모드 True : 자동  False : 수동
        self.flag_down_data = True     #데이터 다운 모드 False : 다운안함 True : 다운
        self.flag_searching = 'Volatility_falling'   #무엇을 백테스팅하는가? 'Volatility_cal'  'Volatility_falling' 'moving ave_60'
                                                    #                   변동성돌파 계산        변동성돌파 차례로     이동평균선60
        self.flag_buy = False
        self.flag_losscut = False   #loss cut 적용여부
        # self.loss_cut = [x for x in range(1,100,1)]   #loss cut(percent)
        self.loss_cut = [100]
        self.margin = False     #마진거래 적용 여부
        self.leverage = [1]     #마진 레버리지 배수
        self.numforma = [x for x in range(1,100,1)]
        # self.numforma = [30]
        if not self.margin:
            self.leverage = [1]
        self.plot = True        #그래프 표시여부
        self.list_trader = ['upbit']    #트레이딩 마켓
        self.trader = pyupbit
        ticker_list = ['BTC']
        self.ticker = {f'KRW-{key}': 1/len(ticker_list) for key in dict.fromkeys(ticker_list).keys()}     #ticker list
        self.K_value = [0.3, 0.6]    #K value
        # self.K_value = [0.3]    #K value
        self.slippage = 0.2
        self.flag_var = [False]
        self.sell_hour = 9
        self.sell_minute = 20
        self.start_time = datetime.datetime(2021, 1, 1, 9)
        self.end_time = datetime.datetime(2022, 5, 15, 9)
        self.temp_result = pd.DataFrame([('binance','BTC/KRW',0.0,False,0,0.0,0.0)], columns=self.column_result)
        for year in range(self.start_time.year, self.end_time.year, 1):
            #print(year)
            str_year = f'{year}'
            self.temp_result[str_year] = 0
        # minute1/minute3/minute5/minute10/minute15/minute30/minute60/minute240/day/week/month
        self.period = 'minute10'  #데이터 다운로드 period

        self.temp_result['now']=0
        #print(self.temp_result)
        self.temp_result.drop([0], inplace=True)

        for tradername in self.list_trader:
            # flag_mode == True시 조건에 맞는 ticker를 필터링하여 self.ticker에 저장
            # flag_mode == fasle시 __init__에 설정된 self.ticker 사용
            print(len(self.ticker),'\n',self.ticker)
            ll = []
            self.temp_result = pd.DataFrame()
            # self.temp_result = self.temp_result.append(pd.Series(dtype=float, name=0))
            # print('temp result',self.temp_result)
            #ticker list를 돌아가며 검색
            for ticker in self.ticker.keys():
                try:
                    # ticker별 ohlcv 데이터 다운로드
                    ticker1 = ticker.split('-')[1]
                    ticker2 = ticker.split('-')[0]
                    path = f'./data/{self.period}_{ticker1}_{ticker2}_{self.start_time.timestamp()}_{self.end_time.timestamp()}.csv'
                    if self.flag_down_data: #데이터를 직접 다운로드
                        data = func.datadown(self.trader, self.start_time, self.end_time, ticker, self.period, self.margin, path)
                    else:  # 기존 저장된 데이터를 불러옴
                        try:
                            data = pd.read_csv(path)
                            data.set_index(keys='timestamp', inplace=True)
                        except Exception as e:
                            print('no saved data')
                            data = func.datadown(self.trader, self.start_time, self.end_time, ticker, self.period, self.margin, path)
                    data.reset_index(inplace=True)
                    data.sort_values(by='timestamp', axis=0, inplace=True)
                    data.reset_index(drop=True, inplace=True)

                    #계산
                    if self.flag_searching == 'Volatility_cal':  #calculate Dataframe
                        for cut_k in self.K_value:
                            for var in self.flag_var:
                                for lev in self.leverage:
                                    pre_hpr = 1
                                    temp_data = data.copy(False)
                                    temp_data['range'] = (temp_data['high'] - temp_data['low']) * float(cut_k)
                                    temp_data['target'] = temp_data['open'] + temp_data['range'].shift(1)
                                    if self.margin:
                                        temp_data['ror'] = np.where(temp_data['high'] > temp_data['target'],
                                                            np.where((temp_data['close'] - temp_data['target']) * lev/temp_data['target'] + 1 < 0, 0, (temp_data['close'] - temp_data['target']) * lev/temp_data['target']+1-self.slippage/100) ,
                                                             1)
                                    else:
                                        temp_data['ror'] = np.where(temp_data['high'] > temp_data['target'],
                                                            (temp_data['close'] - temp_data['target']) / temp_data['target'] + 1 - self.slippage / 100,
                                                                    1)
                                    temp_data['ticker'] = ticker
                                    ll.append(temp_data)

                                    temp_data['hpr'] = temp_data['ror'].cumprod()
                                    temp_data['dd'] = (temp_data['hpr'].cummax() - temp_data['hpr']) / temp_data['hpr'].cummax() * 100

                                    print("\nticker : ",ticker)
                                    print("K value : ",cut_k)
                                    print("hrp : ",temp_data.iloc[-1]['hpr'])
                                    print("MDD(%): ", temp_data['dd'].max(),"\n")

                                    result = pd.DataFrame([(tradername, ticker, cut_k, var, lev, temp_data.iloc[-1]['hpr'], temp_data['dd'].max())], columns=self.column_result)

                                    #ticker별 결과 저장
                                    ticker1 = ticker.split('-')[1]
                                    ticker2 = ticker.split('-')[0]
                                    temp_path = f'./result/detail/d_result_{ticker1}_{ticker2}_{cut_k}_{var}_{lev}.xlsx'
                                    temp_data.to_excel(temp_path)

                                    #년도별 수익률 계산 및 저장
                                    for year in range(self.start_time.year, self.end_time.year, 1):
                                        #print(year)
                                        str_year = f'{year}'
                                        for i in range(len(temp_data)):
                                            if temp_data.iloc[i]['time'].year == year and temp_data.iloc[i]['time'].month == 12 and temp_data.iloc[i]['time'].day == 31:
                                                result.loc[0, str_year] = temp_data.iloc[i]['hpr'] / pre_hpr
                                                pre_hpr = temp_data.iloc[i]['hpr']
                                                break
                                    result.loc[0,'now'] = temp_data.iloc[-1]['hpr']/pre_hpr
                                    self.temp_result = pd.concat([self.temp_result, result],ignore_index=True)
                                    print('print',self.temp_result)

                                    #그래프 저장
                                    if self.plot:
                                        plt.figure(figsize=(10, 10))
                                        plt.plot(temp_data['time'], temp_data['hpr'], 'b-', label='yield')
                                        plt.plot(temp_data['time'], temp_data['ratio'], 'r-', label=ticker)
                                        title = ticker
                                        plt.title(title)
                                        plt.legend(loc='best')
                                        plt.grid(True)
                                        ticker1 = ticker.split('-')[1]
                                        ticker2 = ticker.split('-')[0]
                                        temp_path = f'./graph/graph_{ticker1}_{ticker2}_{cut_k}_{var}_{lev}.png'
                                        plt.savefig(temp_path)
                                        plt.clf()
                                        #plt.show()

                    elif self.flag_searching == 'Volatility_falling':   #calculate falling data
                        print('Volatility_falling')
                        for cut_k in self.K_value:
                            for var in self.flag_var:
                                for lev in self.leverage:
                                    for loss in self.loss_cut:
                                        pre_hpr = 1
                                        pre_timestamp = 0
                                        flag_tageted = False
                                        list_ror = []
                                        data_ror = pd.DataFrame([(0, 0, 0, 0, 0, 0, 0, 0)], columns=self.column_data)
                                        data_ror.drop([0], inplace=True)
                                        temp_data = data.copy(False)
                                        print(temp_data)
                                        for i in range(0,len(temp_data),1):
                                            now_data = temp_data.iloc[i]
                                            now_time = datetime.datetime.fromtimestamp(now_data['timestamp'])
                                            print(now_time,now_data['time'])
                                            try:
                                                if now_time.hour == 9 and now_time.minute == 0:  #9시가 되었을때
                                                    # print(now_time)
                                                    if pre_timestamp > 0:
                                                        open = now_data['open']
                                                        pre_target = target
                                                        target = open + (high - low) * cut_k
                                                        target_loss = target * (1-loss/100)
                                                        pre_timestamp = now_data['timestamp']
                                                        pre_open = open
                                                        volume = now_data['volume']
                                                        high = now_data['high']
                                                        low = now_data['low']
                                                    else:
                                                        print(i,now_time)
                                                        raise Exception('first day ~~')

                                                if pre_timestamp == 0:
                                                    continue
                                                volume += now_data['volume']

                                                if now_time.hour == self.sell_hour and now_time.minute == self.sell_minute:  #sell time
                                                    # print(now_time)
                                                    close = temp_data.iloc[i - 1]['close']
                                                    if self.flag_buy:
                                                        if self.flag_losscut:
                                                            self.flag_losscut = False
                                                        else:
                                                            price_sell = close
                                                            self.flag_buy = False
                                                        ror = 1 + (price_sell - price_buy) / price_buy - self.slippage / 100
                                                    else:
                                                        ror = 1
                                                    try:
                                                        temp = pd.DataFrame([(pre_timestamp, close, price_buy, low, close, target, now_time - datetime.timedelta(days=1), now_data['ratio'])], columns=self.column_data)
                                                    except:
                                                        temp = pd.DataFrame([(pre_timestamp, pre_open, high, low, close, target, now_time - datetime.timedelta(days=1), now_data['ratio'])], columns=self.column_data)

                                                    data_ror = data_ror.append(temp, ignore_index=True)
                                                    list_ror.append(ror)
                                                    flag_tageted = True

                                                if self.flag_buy and now_data['low'] < target_loss and self.flag_buy:  # loss cut 발생시
                                                    price_sell = target_loss
                                                    self.flag_losscut = True

                                                if flag_tageted:    # 사야할때
                                                    # if datetime.datetime(year=2022, month=1, day=1, hour=self.sell_hour, minute=self.sell_minute) > datetime.datetime(year = 2022,month = 1,day = 1,hour=9,minute=0): #이미 sell time에 target 완료했으면 buy 처리
                                                    #     if temp_data.iloc[i - 1]['close'] > target and not target == 0:
                                                    #         price_buy = target
                                                    #         self.flag_buy = True
                                                    if now_data['high'] > target and not target == 0:   #목표가 달성시 flag_buy 올린다
                                                        price_buy = target
                                                        self.flag_buy = True

                                                #high와 low 값 정의
                                                if now_data['high'] > high:
                                                    high = now_data['high']
                                                if now_data['low'] < low:
                                                    low = now_data['low']

                                            except Exception as e:  # 첫날 실행시 이전 데이터 저장
                                                open = now_data['open']
                                                pre_timestamp = now_data['timestamp']
                                                pre_open = open
                                                volume = now_data['volume']
                                                high = now_data['high']
                                                low = now_data['low']
                                                self.flag_buy = False
                                                target = 0
                                                print(e)
                                        #수익률 계산
                                        print('calculate start')
                                        data_ror['ror'] = list_ror
                                        data_ror['ticker'] = ticker
                                        ll.append(data_ror)
                                        data_ror['hpr'] = data_ror['ror'].cumprod()
                                        data_ror['dd'] = (data_ror['hpr'].cummax() - data_ror['hpr']) / data_ror[
                                            'hpr'].cummax() * 100

                                        print("\nticker : ", ticker)
                                        print("K value : ", cut_k)
                                        print("hrp : ", data_ror.iloc[-1]['hpr'])
                                        print("MDD(%): ", data_ror['dd'].max(), "\n")

                                        result = pd.DataFrame([(tradername, ticker, cut_k, var, lev, loss,
                                                                data_ror.iloc[-1]['hpr'], data_ror['dd'].max())],
                                                              columns=self.column_result_falling)
                                        #detail data 저장
                                        ticker1 = ticker.split('-')[1]
                                        ticker2 = ticker.split('-')[0]
                                        temp_path = f'./result/detail/d_result_{ticker1}_{ticker2}_{cut_k}_{var}_{lev}_{loss}_falling.xlsx'
                                        data_ror.to_excel(temp_path)

                                        # 년도별 수익률 계산 및 저장
                                        for year in range(self.start_time.year, self.end_time.year, 1):
                                            str_year = f'{year}'
                                            for i in range(len(data_ror)):
                                                # if data_ror.iloc[i]['time'].year == year and data_ror.iloc[i]['time'].month == 12 and data_ror.iloc[i]['time'].day == 31:
                                                if data_ror.iloc[i]['time'].year == year+1:
                                                    print(f'{data_ror.iloc[i]["time"].year} {year}')
                                                    result.loc[0, str_year] = data_ror.iloc[i-1]['hpr'] / pre_hpr
                                                    pre_hpr = data_ror.iloc[i-1]['hpr']
                                                    break
                                        result.loc[0, 'now'] = data_ror.iloc[-1]['hpr'] / pre_hpr
                                        self.temp_result = pd.concat([self.temp_result, result], ignore_index=True)
                                        print('print',self.temp_result)

                                        # 그래프 저장
                                        if self.plot:
                                            plt.figure(figsize=(10, 10))
                                            plt.plot(data_ror['time'], data_ror['hpr'], 'b-', label='yield')
                                            plt.plot(data_ror['time'], data_ror['ratio'], 'r-', label=ticker)
                                            title = ticker
                                            plt.title(title)
                                            plt.legend(loc='best')
                                            plt.grid(True)
                                            ticker1 = ticker.split('-')[1]
                                            ticker2 = ticker.split('-')[0]
                                            temp_path = f'./graph/graph_{ticker1}_{ticker2}_{cut_k}_{var}_{lev}_falling.png'
                                            plt.savefig(temp_path)
                                            plt.clf()
                                            # plt.show()
                    elif self.flag_searching == 'moving ave_60':
                        for ma in self.numforma:
                            for loss in self.loss_cut:
                                flagflag = False
                                flag1 = False
                                price_buy = 0
                                price_sell = 0
                                price_loss = 0
                                pre_hpr = 1
                                list_ror = [1]
                                list_buy = [0]
                                list_sell = [0]
                                print(f'moving ave_{ma} {loss}')
                                temp_data = data.copy(False)
                                temp_data[f'sma'] = temp_data['close'].rolling(ma).mean()
                                temp_data = temp_data[ma-1:].reset_index(drop=True)
                                temp_data['ratio'] = temp_data['ratio'] / temp_data['ratio'][0]
                                for ind in range(1,len(temp_data)):
                                    pre = temp_data.iloc[ind-1]
                                    now = temp_data.iloc[ind]
                                    if ind == 1:
                                        if now['high'] >= pre['sma']:
                                            flag1 = True
                                            flagflag = True
                                    if flagflag:
                                        if now['low'] < price_loss:
                                            if flag1:
                                                list_ror.append(1)
                                                list_buy.append(price_buy)
                                                list_sell.append(price_sell)
                                                flag1 = False
                                                flagflag = False
                                            else:
                                                price_sell = min([price_loss,now['open']])
                                                list_ror.append(price_sell / price_buy-self.slippage)
                                                list_buy.append(price_buy)
                                                list_sell.append(price_sell)
                                                price_buy = 0
                                                price_sell = 0
                                                price_loss = 0
                                                flagflag = False
                                        elif now['low'] < pre['sma']:
                                            if flag1:
                                                list_ror.append(1)
                                                list_buy.append(price_buy)
                                                list_sell.append(price_sell)
                                                flag1 = False
                                                flagflag = False
                                            else:
                                                price_sell = min([pre['sma'],now['open']])
                                                list_ror.append(price_sell/price_buy-self.slippage)
                                                list_buy.append(price_buy)
                                                list_sell.append(price_sell)
                                                price_buy = 0
                                                price_sell = 0
                                                price_loss = 0
                                                flagflag = False
                                        else:
                                            list_ror.append(1)
                                            list_buy.append(price_buy)
                                            list_sell.append(price_sell)
                                    else:
                                        list_ror.append(1)
                                        list_buy.append(price_buy)
                                        list_sell.append(price_sell)
                                        if now['high'] >= pre['sma']:
                                            price_buy = max([pre['sma'],now['open']])
                                            # price_buy = pre['sma']    # 설레게한 치명적 오류
                                            price_loss = price_buy*(1-loss/100)
                                            flagflag = True
                                print(len(temp_data),' ',len(list_ror),' ',len(list_buy),' ',len(list_sell))
                                print('calculate start')
                                temp_data['price buy'] = list_buy
                                temp_data['price sell'] = list_sell
                                temp_data['ror'] = list_ror
                                temp_data['ticker'] = ticker
                                ll.append(temp_data)
                                temp_data['hpr'] = temp_data['ror'].cumprod()
                                temp_data['dd'] = (temp_data['hpr'].cummax() - temp_data['hpr']) / temp_data['hpr'].cummax() * 100

                                print("\nticker : ", ticker)
                                print("hrp : ", temp_data.iloc[-1]['hpr'])
                                print("MDD(%): ", temp_data['dd'].max(), "\n")

                                result = pd.DataFrame([(tradername, ticker, ma,loss, temp_data.iloc[-1]['hpr'],
                                                        temp_data['dd'].max())], columns=['market','ticker','ma','losscut','HPR','MDD'])

                                # ticker별 결과 저장
                                ticker1 = ticker.split('-')[1]
                                ticker2 = ticker.split('-')[0]
                                temp_path = f'./result/detail/d_result_{ticker1}_{ticker2}_ma{ma}_{loss}.csv'
                                temp_data.to_csv(temp_path)

                                # 년도별 수익률 계산 및 저장
                                for year in range(self.start_time.year, self.end_time.year, 1):
                                    # print(year)
                                    str_year = f'{year}'
                                    for i in range(len(temp_data)):
                                        if temp_data.iloc[i]['time'].year == year and temp_data.iloc[i]['time'].month == 12 and temp_data.iloc[i]['time'].day == 31:
                                            result.loc[0, str_year] = temp_data.iloc[i]['hpr'] / pre_hpr
                                            pre_hpr = temp_data.iloc[i]['hpr']
                                            break
                                result.loc[0, 'now'] = temp_data.iloc[-1]['hpr'] / pre_hpr
                                self.temp_result = self.temp_result.append(result, ignore_index=True)

                                # 그래프 저장
                                if self.plot:
                                    plt.figure(figsize=(10, 10))
                                    fig, ax1 = plt.subplots()

                                    ax1.plot(temp_data['time'], temp_data['sma'], 'g-', label='sma')
                                    ax1.plot(temp_data['time'], temp_data['close'], 'r-', label=ticker)
                                    ax2 = ax1.twinx()
                                    ax2.plot(temp_data['time'], temp_data['hpr'], 'b-', label='yield')

                                    title = ticker
                                    plt.title(title)
                                    plt.legend(loc='best')
                                    plt.grid(True)
                                    ticker1 = ticker.split('-')[1]
                                    ticker2 = ticker.split('-')[0]
                                    temp_path = f'./graph/graph_{ticker1}_{ticker2}_ma{ma}_{loss}.png'
                                    plt.savefig(temp_path)
                                    plt.clf()
                except Exception as e:
                    print(e)

            self.temp_result.to_excel(f'./result/result_{self.end_time.year}_{self.end_time.month}_{self.end_time.day}_sum.xlsx')
        #
        #     # 각 티커 분산투자로 인한 데이터 합 계산
        #     if True:
        #         pre_hpr = 1
        #         final_data = pd.DataFrame()
        #         print(ll)
        #         for x in range(len(ll)):
        #             if x == 0:
        #                 final_data['time'] = ll[x]['time']
        #                 final_data['ratio'] = ll[x]['ratio']
        #                 final_data[f"ror_{x}_{ll[x]['ticker'][0].split(-)[1]}_{float(self.ticker[ll[x]['ticker'][0].split('-')[1]])}"] = ll[x]['ror']
        #
        #             else:
        #                 second_data = pd.DataFrame()
        #                 second_data['time'] = ll[x]['time']
        #                 second_data[f"ror_{x}_{ll[x]['ticker'][0].split('-')[1]}_{float(self.ticker[ll[x]['ticker'][0].split('-')[1]])}"] = ll[x]['ror']
        #                 final_data = pd.merge(final_data,second_data,on='time',how='outer')
        #         final_data.replace(np.NaN,1,inplace=True)
        #         final_data['final_ror'] = 0
        #         for col in final_data.columns:
        #             try:
        #                 print(col)
        #                 if 'ror' in col:
        #                     print(col.split('_')[-1])
        #                     multiple = float(col.split('_')[-1])
        #                     print('multiple',multiple)
        #                     final_data['final_ror'] += multiple * final_data[col]
        #             except Exception as e:
        #                 print(e)
        #
        #         print('sum ror',final_data)
        #
        #         final_data['hpr'] = final_data['final_ror'].cumprod()
        #         final_data['dd'] = (final_data['hpr'].cummax() - final_data['hpr']) / final_data['hpr'].cummax() * 100
        #
        #         print("분산투자시 결과")
        #         print("hrp : ", final_data.iloc[-1]['hpr'])
        #         print("MDD(%): ", final_data['dd'].max())
        #
        #         result = pd.DataFrame(
        #             [(tradername, ticker, 'sum', 0, 0, final_data.iloc[-1]['hpr'], final_data['dd'].max())],
        #             columns=self.column_result)
        #         ticker1 = ticker.split('-')[1]
        #         ticker2 = ticker.split('-')[0]
        #         temp_path = f'./result/detail/d_result_{ticker1}_{ticker2}_sum.xlsx'
        #         final_data.to_excel(temp_path)
        #
        #         # 년도별 수익률 계산 및 저장
        #         for year in range(self.start_time.year, self.end_time.year, 1):
        #             # print(year)
        #             str_year = f'{year}'
        #             for i in range(0, len(final_data), 1):
        #                 if final_data.iloc[i]['time'].year == year and final_data.iloc[i]['time'].month == 12 and \
        #                         final_data.iloc[i]['time'].day == 31:
        #                     result.loc[0, str_year] = final_data.iloc[i]['hpr'] / pre_hpr
        #                     pre_hpr = final_data.iloc[i]['hpr']
        #                     # print(final_data.iloc[i]['time'])
        #                     break
        #         result.loc[0, 'now'] = final_data.iloc[-1]['hpr'] / pre_hpr
        #         self.temp_result = self.temp_result.append(result, ignore_index=True)
        #
        #         # 그래프 저장
        #         if self.plot:
        #             print('plot')
        #             plt.figure(figsize=(10, 10))
        #             plt.plot(final_data['time'], final_data['hpr'], 'b-', label='yield')
        #             plt.plot(final_data['time'], final_data['ratio'], 'r-', label=ticker)
        #             title = ticker
        #             plt.title(title)
        #             plt.legend(loc='best')
        #             plt.grid(True)
        #             ticker1 = ticker.split('-')[1]
        #             ticker2 = ticker.split('-')[0]
        #             temp_path = f'./graph/graph_{ticker1}_{ticker2}_sum.png'
        #             plt.savefig(temp_path)
        #             plt.clf()
        #             # plt.show()
        # print(self.temp_result)
        # self.temp_result.to_excel(f'./result/result_{self.end_time.year}_{self.end_time.month}_{self.end_time.day}_sum.xlsx')


    def pick_ticker(self,tradername, trader):   #조건에 맞는 ticker를 필터링하는 함수
        if self.flag_mode:
            tmp_list = []
            self.ticker1 = trader.fetch_tickers()
            temp = pd.DataFrame(self.ticker1)
            if tradername == 'binance':
                temp = temp.filter(like='/USDT',axis=1)
            elif tradername == 'upbit':
                temp = temp.filter(like='/KRW',axis=1)
            for key in temp.keys():
                if temp[key]['change'] == 0:
                    del temp[key]
            print(len(temp.columns))
            try:
                for i in temp.columns:
                    time.sleep(0.01)
                    print(i)
                    price = self.trader.fetchOHLCV(i,'1w')
                    tmp_p = pd.DataFrame(price,columns=self.column_ohlcv)
                    C_price = tmp_p.iloc[-2]
                    if tradername == 'binance':
                        amount_trading = 100000000
                    elif tradername == 'upbit':
                        amount_trading = 100000000000
                    if C_price['volume']*C_price['close'] < amount_trading:       #거래량 부족분 삭제
                        del temp[i]
                    if i == 'BUSD/USDT':
                        del temp[i]
            except Exception as e:
                print('err here1',e)
            print(len(temp.columns))
            temp_list = []
            for i in temp.columns:
                temp_list.append(i.split('-')[1])
            self.ticker = {key: 1 for key in temp_list}

        elif self.flag_mode == 2:
            print('mode 2')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/