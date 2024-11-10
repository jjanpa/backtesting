import pandas as pd
import datetime
import time
import pyupbit
import ccxt

def datadown(trader_name, start_time, end_time, ticker, period, flag_margin, path):  # ticker의 period 간격의 ohlcv 데이터를 저장 및 return
    column_ohlcv = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    try:
        temp_ohlcv = pd.DataFrame()
        print(f'data down {ticker} {period}')
        print(trader_name)
        if trader_name == 'upbit':
            trader = pyupbit
        if trader_name == 'binance':
            trader = ccxt.binance()
        if trader_name == 'binance_m':
            trader = ccxt.binance({'options': {'defaultType': 'future'},})

        # delta 설정
        # minute1/minute3/minute5/minute10/minute15/minute30/minute60/minute240/day/week/month
        if period == 'minute1':
            delta = datetime.timedelta(minutes=1)
        elif period == 'minute3':
            delta = datetime.timedelta(minutes=3)
        elif period == 'minute5':
            delta = datetime.timedelta(minutes=5)
        elif period == 'minute10':
            delta = datetime.timedelta(minutes=10)
        elif period == 'minute15':
            delta = datetime.timedelta(minutes=15)
        elif period == 'minute30':
            delta = datetime.timedelta(minutes=30)
        elif period == 'minute60':
            delta = datetime.timedelta(minutes=60)
        elif period == 'minute240':
            delta = datetime.timedelta(minutes=240)
        elif period == 'day':
            delta = datetime.timedelta(days=1)
        elif period == 'week':
            delta = datetime.timedelta(weeks=1)
        if trader_name == 'upbit':
            n_time = end_time
            print(ticker)
            print('...end time ', n_time)
            num_data = 0
            # end_time까지 temp_ohlcv에 데이터 다운
            while start_time <= n_time:
                try:
                    print(f'data down {n_time}')
                    time.sleep(0.05)
                    ohlcv = trader.get_ohlcv(ticker, to=n_time, interval=period, count=1000)
                    # ohlcv = pd.DataFrame(ohlcv, columns=column_ohlcv)
                    if len(ohlcv) != 0:
                        if len(temp_ohlcv) == 0:
                            temp_ohlcv = ohlcv
                            n_time = temp_ohlcv.index[0]
                        else:
                            temp_ohlcv = pd.concat([temp_ohlcv, ohlcv])
                            temp_ohlcv.sort_index(inplace=True)
                            n_time = temp_ohlcv.index[0]
                    else:
                        n_time = n_time - delta * 200
                except Exception as e:
                    print('err data down loop', e)
                    break
            temp_ohlcv.sort_index(inplace=True)
            temp_ohlcv.rename_axis('time', inplace=True)
            temp_ohlcv.reset_index(inplace=True)
            start_price = temp_ohlcv.iloc[0]['close']
            list_timestamp = []
            list_ratio = []
            ### 시간,Ratio 정보 저장
            for i in range(len(temp_ohlcv)):
                list_timestamp.append(datetime.datetime.timestamp(temp_ohlcv.iloc[i]['time']))
                list_ratio.append(temp_ohlcv.iloc[i]['close'] / start_price)
            temp_ohlcv['timestamp'] = list_timestamp
            temp_ohlcv['ratio'] = list_ratio
            temp_ohlcv.set_index(keys='timestamp', inplace=True)
        elif trader_name == 'binance' or trader_name == 'binance_m':
            n_time = start_time
            print('...start time ', n_time)
            num_data = 0
            try:
                #end_time까지 temp_ohlcv에 데이터 다운
                while end_time >= n_time:
                    print(f'data down {n_time}')
                    time.sleep(0.05)
                    ohlcv = trader.fetchOHLCV('ETH/USDT', '1h', int(n_time.timestamp())*1000)
                    ohlcv = pd.DataFrame(ohlcv,columns=column_ohlcv)
                    if len(ohlcv)!=0:
                        temp_ohlcv = temp_ohlcv.append(ohlcv,ignore_index=True)
                        n_time = datetime.datetime.fromtimestamp(int(ohlcv.iloc[-1]['timestamp'])/1000)
                    else:
                        n_time = n_time + delta *500
            except Exception as e:
                print('err',e)
            start_price = temp_ohlcv.iloc[0]['close']
            temp_ohlcv['timestamp'] = temp_ohlcv['timestamp']/1000
            list_time = []
            list_ratio = []
            ### 시간,Ratio 정보 저장
            for i in range(len(temp_ohlcv)):
                list_time.append(datetime.datetime.fromtimestamp(temp_ohlcv.iloc[i]['timestamp']))
                list_ratio.append(temp_ohlcv.iloc[i]['close'] / start_price)
            temp_ohlcv['time'] = list_time
            temp_ohlcv['ratio'] = list_ratio
            temp_ohlcv.set_index(keys='timestamp', inplace=True)
        try:
            temp_ohlcv = temp_ohlcv[int(start_time.timestamp()):]
        except Exception as e:
            print('시작점을 못잘랐따리~')
        print('make data done')
        temp_ohlcv.to_csv(path)
        print('save csv data')
        print(f'data down done')
        print(temp_ohlcv)
        return temp_ohlcv
    except Exception as e:
        print(f'data down fail : {e}')
