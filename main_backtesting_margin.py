import numpy as np
import sys
from PyQt5.QtWidgets import *
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ccxt

class main():
    def __init__(self):
        super().__init__()
        ########## 변수 설정
        self.flag_mode = False   #데이터 검색 모드 True : 자동  False : 수동
        self.flag_down_data = False     #데이터 다운 모드 False : 다운안함 True : 다운
        self.flag_seaching = False      #데이터 처리 방식 True : calculating False : falling
        self.flag_buy = False
        self.flag_losscut = False   #loss cut 적용여부
        self.loss_cut = [100]   #loss cut(percent)
        self.margin = False     #마진거래 적용 여부
        self.leverage = [5]     #마진 레버리지 배수
        if not self.margin:
            self.leverage = [1]
        self.plot = True        #그래프 표시여부
        self.list_trader = ['upbit']    #트레이딩 마켓
        self.ticker = {'BTC':1/12,'ETH':1/12}     #ticker list
        self.num_maxdata = 100000   #엑셀데이터 최대 저장 크기
        self.K_value = [0.3,0.4,0.5,0.6,0.7,0.8]    #K value
        self.slippage = 0.2
        self.flag_var = [False]
        #self.start_time = int(datetime.datetime(2021,5,1,9).timestamp())*1000
        self.sell_hour = 10
        self.sell_minute = 0
        temp_time = datetime.datetime(2022,2,20,9)
        self.end_time = int(temp_time.timestamp())*1000
        self.start_time = int(datetime.datetime(2016,1,1,9).timestamp())*1000
        # self.start_time = int(datetime.datetime(temp_time.year,temp_time.month-2,temp_time.day,temp_time.hour).timestamp())*1000
        if self.flag_seaching:
            self.period = '1d'
        else:
            self.period = '30m'

        ### DataFrame 선언
        self.column_ohlcv = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.column_data = ['timestamp','open', 'high', 'low', 'close', 'volume','time','ratio']
        self.column_result = ['trader', 'ticker', 'k','var','leverage','hpr','MDD']
        self.column_result_falling = ['trader', 'ticker', 'k','var','leverage','losscut','hpr','MDD']

        self.temp_result = pd.DataFrame([('binance','BTC/KRW',0.0,False,0,0.0,0.0)], columns=self.column_result)
        for year in range(datetime.datetime.fromtimestamp(self.start_time/1000).year,datetime.datetime.fromtimestamp(self.end_time/1000).year,1):
            #print(year)
            str_year = f'{year}'
            self.temp_result[str_year] = 0
        self.temp_result['now']=0
        #print(self.temp_result)
        self.temp_result.drop([0], inplace=True)

        for tradername in self.list_trader:
            #마켓 ccxt trader 설정
            if tradername == 'binance':
                if self.margin:
                    print('margin')
                    self.trader = ccxt.binance({
                        'enableRateLimit': True,
                        'options': {
                            'defaultType': 'future'
                        }
                    })
                else:
                    self.trader = ccxt.binance()
            else:
                self.trader = ccxt.upbit()

            # flag_mode == True시 조건에 맞는 ticker를 필터링하여 self.ticker에 저장
            # flag_mode == fasle시 __init__에 설정된 self.ticker 사용
            self.pick_ticker(tradername, self.trader)
            ll = []
            #ticker list를 돌아가며 검색
            for ticker in self.ticker.keys():
                try:
                    if tradername == 'binance':
                        ticker = ticker + '/USDT'
                    else:
                        ticker = ticker + '/KRW'

                    #ticker별 ohlcv 데이터 다운로드
                    if self.flag_down_data: #데이터를 직접 다운로드
                        data = self.datadown(tradername, ticker, self.period)
                    else:   #기존 저장된 데이터를 불러옴
                        data = pd.DataFrame([(0, 0, 0, 0, 0, 0)], columns=self.column_ohlcv)
                        data.drop([0], inplace=True)
                        num_data = 0
                        ticker1 = ticker.split('/')[0]
                        ticker2 = ticker.split('/')[1]
                        while True:
                            try:
                                if self.margin:
                                    temp_path = f'./data/margin_{self.period}_{ticker1}_{ticker2}_{num_data}.xlsx'
                                else:
                                    temp_path = f'./data/{self.period}_{ticker1}_{ticker2}_{num_data}.xlsx'
                                data = data.append(pd.read_excel(temp_path), ignore_index=True)
                                num_data += 1
                            except Exception as e:
                                print('data loading done')
                                break
                    data.sort_values(by='timestamp', axis=0, inplace=True)

                    #계산
                    if self.flag_seaching:  #calculate Dataframe
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
                                    ticker1 = ticker.split('/')[0]
                                    ticker2 = ticker.split('/')[1]
                                    temp_path = f'./result/detail/d_result_{ticker1}_{ticker2}_{cut_k}_{var}_{lev}.xlsx'
                                    temp_data.to_excel(temp_path)

                                    #년도별 수익률 계산 및 저장
                                    for year in range(datetime.datetime.fromtimestamp(self.start_time / 1000).year,
                                                      datetime.datetime.fromtimestamp(self.end_time / 1000).year, 1):
                                        #print(year)
                                        str_year = f'{year}'
                                        for i in range(len(temp_data)):
                                            if temp_data.iloc[i]['time'].year == year and temp_data.iloc[i]['time'].month == 12 and temp_data.iloc[i]['time'].day == 31:
                                                result.loc[0, str_year] = temp_data.iloc[i]['hpr'] / pre_hpr
                                                pre_hpr = temp_data.iloc[i]['hpr']
                                                break
                                    result.loc[0,'now'] = temp_data.iloc[-1]['hpr']/pre_hpr
                                    self.temp_result = self.temp_result.append(result,ignore_index=True)

                                    #그래프 저장
                                    if self.plot:
                                        plt.figure(figsize=(10, 10))
                                        plt.plot(temp_data['time'], temp_data['hpr'], 'b-', label='yield')
                                        plt.plot(temp_data['time'], temp_data['ratio'], 'r-', label=ticker)
                                        title = ticker
                                        plt.title(title)
                                        plt.legend(loc='best')
                                        plt.grid(True)
                                        ticker1 = ticker.split('/')[0]
                                        ticker2 = ticker.split('/')[1]
                                        temp_path = f'./graph/graph_{ticker1}_{ticker2}_{cut_k}_{var}_{lev}.png'
                                        plt.savefig(temp_path)
                                        #plt.show()



                    else:   #calculate falling data
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
                                        for i in range(len(temp_data)):
                                            now_data = temp_data.iloc[i]
                                            try:
                                                if now_data['time'].hour == 9 and now_data['time'].minute == 0:  #9시가 되었을때
                                                    if pre_timestamp > 0:
                                                        open = now_data['open']
                                                        pre_target = target
                                                        target = open + (high - low) * cut_k
                                                        target_loss = target * (1-loss/100)
                                                        pre_timestamp = now_data['timestamp']
                                                        pre_open = open
                                                        volume = 0
                                                        high = now_data['high']
                                                        low = now_data['low']
                                                    else:
                                                        raise Exception('first day ~~')

                                                volume += now_data['volume']

                                                if now_data['time'].hour == self.sell_hour and now_data['time'].minute == self.sell_minute:  #sell time
                                                    close = temp_data.iloc[i - 1]['close']
                                                    if self.flag_buy:
                                                        if self.flag_losscut:
                                                            self.flag_losscut = False
                                                        else:
                                                            price_sell = close
                                                            self.flag_buy = False
                                                        ror = 1 + (price_sell - price_buy) * lev / price_buy - self.slippage / 100
                                                    else:
                                                        ror = 1
                                                    try:
                                                        temp = pd.DataFrame([(pre_timestamp,close, price_buy, low, close, target, now_data['time'] - datetime.timedelta(days=1), now_data['ratio'])], columns=self.column_data)
                                                    except:
                                                        temp = pd.DataFrame([(pre_timestamp, pre_open, high, low, close, target, now_data['time'] - datetime.timedelta(days=1), now_data['ratio'])], columns=self.column_data)

                                                    data_ror = data_ror.append(temp, ignore_index=True)
                                                    list_ror.append(ror)
                                                    flag_tageted = True

                                                if self.flag_buy and now_data['low'] < target_loss and self.flag_buy:  # loss cut 발생시
                                                    price_sell = target_loss
                                                    self.flag_losscut = True

                                                if flag_tageted:    # 사야할때
                                                    if datetime.datetime(year = 2022,month = 1,day = 1,hour=self.sell_hour,minute=self.sell_minute) > datetime.datetime(year = 2022,month = 1,day = 1,hour=9,minute=0): #이미 sell time에 target 완료했으면 buy 처리
                                                        if temp_data.iloc[i - 1]['close'] > target and not target == 0:
                                                            price_buy = target
                                                            self.flag_buy = True
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
                                                volume = 0
                                                high = now_data['high']
                                                low = now_data['low']
                                                self.flag_buy = False
                                                target = 0
                                                print(e)
                                        #수익률 계산
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
                                        ticker1 = ticker.split('/')[0]
                                        ticker2 = ticker.split('/')[1]
                                        temp_path = f'./result/detail/d_result_{ticker1}_{ticker2}_{cut_k}_{var}_{lev}_{loss}_falling.xlsx'
                                        data_ror.to_excel(temp_path)

                                        # 년도별 수익률 계산 및 저장
                                        for year in range(datetime.datetime.fromtimestamp(self.start_time / 1000).year,
                                                          datetime.datetime.fromtimestamp(self.end_time / 1000).year, 1):
                                            str_year = f'{year}'
                                            for i in range(len(data_ror)):
                                                if data_ror.iloc[i]['time'].year == year and data_ror.iloc[i]['time'].month == 12 and data_ror.iloc[i]['time'].day == 31:
                                                    result.loc[0, str_year] = data_ror.iloc[i]['hpr'] / pre_hpr
                                                    pre_hpr = data_ror.iloc[i]['hpr']
                                                    break
                                        result.loc[0, 'now'] = data_ror.iloc[-1]['hpr'] / pre_hpr
                                        self.temp_result = self.temp_result.append(result, ignore_index=True)

                                        # 그래프 저장
                                        if self.plot:
                                            plt.figure(figsize=(10, 10))
                                            plt.plot(data_ror['time'], data_ror['hpr'], 'b-', label='yield')
                                            plt.plot(data_ror['time'], data_ror['ratio'], 'r-', label=ticker)
                                            title = ticker
                                            plt.title(title)
                                            plt.legend(loc='best')
                                            plt.grid(True)
                                            ticker1 = ticker.split('/')[0]
                                            ticker2 = ticker.split('/')[1]
                                            temp_path = f'./graph/graph_{ticker1}_{ticker2}_{cut_k}_{var}_{lev}_falling.png'
                                            plt.savefig(temp_path)
                                            plt.clf()
                                            # plt.show()
                except Exception as e:
                    print(e)

            # 각 티커 분산투자로 인한 데이터 합 계산
            if True:
                pre_hpr = 1
                final_data = pd.DataFrame()
                print(ll)
                for x in range(len(ll)):
                    if x == 0:
                        final_data['time'] = ll[x]['time']
                        final_data['ratio'] = ll[x]['ratio']
                        final_data[f"ror_{x}_{ll[x]['ticker'][0].split('/')[0]}_{float(self.ticker[ll[x]['ticker'][0].split('/')[0]])}"] = ll[x]['ror']

                    else:
                        second_data = pd.DataFrame()
                        second_data['time'] = ll[x]['time']
                        second_data[f"ror_{x}_{ll[x]['ticker'][0].split('/')[0]}_{float(self.ticker[ll[x]['ticker'][0].split('/')[0]])}"] = ll[x]['ror']
                        final_data = pd.merge(final_data,second_data,on='time',how='outer')
                final_data.replace(np.NaN,1,inplace=True)
                final_data['final_ror'] = 0
                for col in final_data.columns:
                    try:
                        print(col)
                        if 'ror' in col:
                            print(col.split('_')[-1])
                            multiple = float(col.split('_')[-1])
                            print('multiple',multiple)
                            final_data['final_ror'] += multiple * final_data[col]
                    except Exception as e:
                        print(e)

                print('sum ror',final_data)

                final_data['hpr'] = final_data['final_ror'].cumprod()
                final_data['dd'] = (final_data['hpr'].cummax() - final_data['hpr']) / final_data['hpr'].cummax() * 100


                print("분산투자시 결과")
                print("hrp : ", final_data.iloc[-1]['hpr'])
                print("MDD(%): ", final_data['dd'].max())

                result = pd.DataFrame(
                    [(tradername, ticker, 'sum', 0, 0, final_data.iloc[-1]['hpr'], final_data['dd'].max())],
                    columns=self.column_result)
                ticker1 = ticker.split('/')[0]
                ticker2 = ticker.split('/')[1]
                temp_path = f'./result/detail/d_result_{ticker1}_{ticker2}_sum.xlsx'
                final_data.to_excel(temp_path)

                # 년도별 수익률 계산 및 저장
                for year in range(datetime.datetime.fromtimestamp(self.start_time / 1000).year,
                                  datetime.datetime.fromtimestamp(self.end_time / 1000).year, 1):
                    # print(year)
                    str_year = f'{year}'
                    for i in range(0, len(final_data), 1):
                        if final_data.iloc[i]['time'].year == year and final_data.iloc[i]['time'].month == 12 and \
                                final_data.iloc[i]['time'].day == 31:
                            result.loc[0, str_year] = final_data.iloc[i]['hpr'] / pre_hpr
                            pre_hpr = final_data.iloc[i]['hpr']
                            # print(final_data.iloc[i]['time'])
                            break
                result.loc[0, 'now'] = final_data.iloc[-1]['hpr'] / pre_hpr
                self.temp_result = self.temp_result.append(result, ignore_index=True)

                # 그래프 저장
                if self.plot:
                    print('plot')
                    plt.figure(figsize=(10, 10))
                    plt.plot(final_data['time'], final_data['hpr'], 'b-', label='yield')
                    plt.plot(final_data['time'], final_data['ratio'], 'r-', label=ticker)
                    title = ticker
                    plt.title(title)
                    plt.legend(loc='best')
                    plt.grid(True)
                    ticker1 = ticker.split('/')[0]
                    ticker2 = ticker.split('/')[1]
                    temp_path = f'./graph/graph_{ticker1}_{ticker2}_sum.png'
                    plt.savefig(temp_path)
                    plt.clf()
                    # plt.show()
        print(self.temp_result)
        temp_time = datetime.datetime.fromtimestamp(self.end_time/1000)
        self.temp_result.to_excel(f'./result/result_{temp_time.year}_{temp_time.month}_{temp_time.day}_sum.xlsx')


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
                temp_list.append(i.split('/')[0])
            self.ticker = {key: 1 for key in temp_list}

        elif self.flag_mode == 2:
            print('mode 2')


    def datadown(self,tradername, ticker, period):  #ticker의 period 간격의 ohlcv 데이터를 저장 및 return
        try:
            temp_ohlcv = pd.DataFrame([(0, 0.0, 0.0, 0.0, 0.0, 0.0)],columns=self.column_ohlcv)
            temp_ohlcv.drop([0],inplace=True)
            print(f'data down {ticker} {period}')

            #delta 설정
            if period == '1m':
                delta = 60*1000
            elif period == '15m':
                delta = 15*60*1000
            elif period == '30m':
                delta = 30*60*1000
            elif period == '1d':
                delta = 24*60*60*1000
            elif period == '3d':
                delta = 3 * 24 * 60 * 60 * 1000
            elif period == '1h':
                delta = 60*60*1000
            elif period == '6h':
                delta = 6*60*60*1000
            elif period == '12h':
                delta = 12*60*60*1000

            n_time = self.start_time
            print('...start time ', datetime.datetime.fromtimestamp(n_time / 1000))
            num_data = 0
            try:
                #end_time까지 temp_ohlcv에 데이터 다운
                while self.end_time >= n_time:
                    print(f'data down {datetime.datetime.fromtimestamp(n_time / 1000)}')
                    time.sleep(0.05)
                    ohlcv = self.trader.fetchOHLCV(ticker, period, n_time)
                    ohlcv = pd.DataFrame(ohlcv,columns=self.column_ohlcv)
                    if len(ohlcv)!=0:
                        temp_ohlcv = temp_ohlcv.append(ohlcv,ignore_index=True)
                        n_time = int(ohlcv.iloc[-1]['timestamp'])
                        n_time = n_time + delta
                    else:
                        n_time = n_time + delta * 500
            except Exception as e:
                print('err',e)

            start_price = temp_ohlcv.loc[0, 'close']
            list_time = []
            list_ratio = []
            len_data = len(temp_ohlcv)
            for i in range(0, len_data, 1): #end_time까지의 데이터만 잘라냄
                # print(temp_ohlcv.iloc[i]['timestamp'], datetime.datetime.fromtimestamp(self.end_time/1000))
                if int(temp_ohlcv.iloc[i]['timestamp']) == self.end_time:
                    temp_ohlcv = temp_ohlcv[:i].copy()
                    break

            ### 시간,Ratio 정보 저장
            for i in range(0, len(temp_ohlcv), 1):
                list_time.append(datetime.datetime.fromtimestamp(temp_ohlcv.loc[i, 'timestamp'] / 1000))
                list_ratio.append(temp_ohlcv.loc[i, 'close'] / start_price)
            temp_ohlcv['time'] = list_time
            temp_ohlcv['ratio'] = list_ratio
            print('make data done')

            temp_save = temp_ohlcv.copy()
            print('copy data done')

            ticker1 = ticker.split('/')[0]
            ticker2 = ticker.split('/')[1]
            temp_save.to_csv(f'./data/{self.period}_{ticker1}_{ticker2}_{self.period}.csv')
            print('save csv data')

            ### 데이터 엑셀 저장 : num_maxdata개씩 잘라서 저장
            while True:
                print('num of data : ',len(temp_save))
                if len(temp_save) > self.num_maxdata:
                    savedata = temp_save[0:self.num_maxdata]
                    if self.margin:
                        temp_path = f'./data/margin_{self.period}_{ticker1}_{ticker2}_{num_data}.xlsx'
                    else:
                        temp_path = f'./data/{self.period}_{ticker1}_{ticker2}_{num_data}.xlsx'
                    #print(temp_path)
                    num_data = num_data + 1
                    savedata.to_excel(temp_path)
                    temp_save = temp_save[self.num_maxdata:].copy() #저장된 데이터 잘라냄
                else:
                    if self.margin:
                        temp_path = f'./data/margin_{self.period}_{ticker1}_{ticker2}_{num_data}.xlsx'
                    else:
                        temp_path = f'./data/{self.period}_{ticker1}_{ticker2}_{num_data}.xlsx'
                        temp_save.to_excel(temp_path)
                    break
            print(f'data down done')
            return temp_ohlcv
        except Exception as e:
            print(f'data down fail : {e}')

# app = QApplication(sys.argv)
# window = MyWindow()
# #window.show()
# app.exec_()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


