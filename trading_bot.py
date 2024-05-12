import pandas as pd
import matplotlib.pyplot as plt
import requests
from time import time
from datetime import datetime
import numpy as np
# import joblib
# from sklearn.preprocessing import RobustScaler

import yfinance as yf


# Fetch historical data from Yahoo Finance
# crypto_data = yf.download('SHIB-EUR', start='2020-01-01', end='2024-05-01')

cryptos = ['BTC', 'ETH', 'ADA','BNB', 'XRP', 'SOL','AVAX','DOT', 'SHIB', 'MATIC', 'DOGE', 'CRO','ATOM','LTC' \
 , 'TRX', 'LINK','FTT','BCH','ALGO','XLM','SHIB','HBAR','FIL', 'VET', 'MKR', 
'RUNE', 'THETA','FTM','FET', 'QNT','BSV','AAVE','NEO','EGLD'] #stx

# cryptos = ['BTC','ETH']

total_transactions = 0
total_wins = 0
total_loss = 0
total_euro = 0 
wallet_btc = 0
total_euro_deposit = 0
total_euro_gain = 0
total_euro_loss = 0

for crypto in cryptos:
    wallet_euro = 100

    ticker = f'{crypto}-EUR'
    # crypto_data = yf.download(ticker, start='2022-01-01', end='2024-05-01')

    crypto_data = yf.download(tickers=ticker, period="1y", interval="1h")
    crypto_data = crypto_data.resample('10H').last()
    # Select only the 'Close' price from the fetched data
    close_price_df = crypto_data['Close']


    percent=[]

    #load the random forest classification model
    filename = 'RF_crypto_model_smart.sav'
    # loaded_model = joblib.load(filename)



    pd.set_option('expand_frame_repr', True)

    K_list = [0,0,0]
    # data = pd.read_csv('btc_data.csv', low_memory=False)
    # #data= data.loc[0,:]
    # data=data.iloc[::-1]
    # #data= data.loc[4000:-1]
    # print(data.head())
    # close_price_df = data['close']
    # time_data = data['date']
    def rsi(last14):
        differences = last14.diff(1).to_numpy()
        differences = np.delete(differences, 0)
        gains = []
        losses = []
        for i in differences:
            if i > 0:
                gains.append(i)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(i))

        av_gain = sum(gains)/len(gains)
        av_loss = sum(losses)/len(losses)

        if av_loss != 0:
            RS=av_gain/av_loss
        else:
            RS = 0
        RSI=100-100/(1+RS)

        return RSI


    def get_sma(prices, rate):
        return prices.rolling(rate).mean()


    def buy(price, wallet_euro, wallet_btc):
        print('you have to buy:')


    def MACD(prices):
        ema26 = prices.ewm(span=26, adjust=False).mean()
        ema12 = prices.ewm(span=12, adjust=False).mean()
        MACD_line = ema26.subtract(ema12)
        ema_of_MACD = MACD_line.ewm(span=9, adjust=False).mean()
        return MACD_line, ema_of_MACD

    def Stochastic(prices,lim):
        prices_list = prices.to_numpy().T
        D=[]
        K_array=[]

        for i in range(0,prices_list.size):
            if i > lim:
                last14 = prices_list[i-lim-1:i+1]
                L14 = min(last14)
                H14 = max(last14)
                last_price = last14[-1]

                K = 100*(last_price-L14)/(H14-L14)
                K_list.pop(0)
                K_list.append(K)
                D.append(sum(K_list)/3)
                K_array.append(K)
            else:
                D.append(0)
                K_array.append(0)
        return D, K_array






    def sell(price, wallet_euro, wallet_btc):

        profit = (price - buy_price)/price*100
        print('profit: ',profit,' %')
        wallet_euro = wallet_euro + wallet_btc * price - 100*0.006
        global total_euro_gain
        global total_euro_loss
        if wallet_euro>100:
            total_euro_gain += abs(100 - wallet_euro)
        else:
            total_euro_loss += abs(100 - wallet_euro)

        wallet_btc = 0
        wallet_euro = 100
        return wallet_btc, wallet_euro


    def is_downtrade(last14sma):
        differences = last14sma.diff(1).to_numpy()
        differences = np.delete(differences, 0)
        gains = []
        losses = []
        abs_differences=[]
        for i in differences:
            abs_differences.append(abs(i))
            if i > 0:
                gains.append(i)
            else:
                losses.append(abs(i))
        average=sum(abs_differences)/len(abs_differences)
        if (len(gains) < len(losses) or 2.5*sum(gains)<sum(losses) or (differences[-1]<-average and differences[-2]<-average)):# and differences[-2]<0) or abs_differences):
            flag = 1
        else:
            flag = 0
        return flag


    def is_uptrend(last14sma):
        differences = last14sma.diff(1).to_numpy()
        differences = np.delete(differences, 0)
        gains = []
        losses = []
        for i in differences:
            if i > 0:
                gains.append(i)
            else:
                losses.append(abs(i))
        # average=sum(losses)/len(losses)

        if (len(gains) > len(losses) or sum(gains)>sum(losses)):
            flag = 1
        else:
            flag = 0
        return flag


    def get_bollinger_bands(prices, rate=20):
        sma = get_sma(prices, rate) # <-- Get SMA for 20 days
        std = prices.rolling(rate).std() # <-- Get rolling standard deviation for 20 days
        bollinger_up = sma + std * 2  # Calculate top band
        bollinger_down = sma - std * 2  # Calculate bottom band
        return bollinger_up, bollinger_down,sma


    def get_hist_data(from_sym='BTC', to_sym='EUR', timeframe='minute', limit=2000, aggregation=1, exchange='', until_time=time()):
        url = 'https://min-api.cryptocompare.com/data/v2/histo'
        url += timeframe
        parameters = {'fsym': from_sym,
                    'tsym': to_sym,
                    'limit': limit,
                    'aggregate': aggregation,
                    'toTs': until_time}
        if exchange:
            print('exchange: ', exchange)
            parameters['e'] = exchange

        print('baseurl: ', url)
        print('timeframe: ', timeframe)
        print('parameters: ', parameters)

        # response comes as json
        response = requests.get(url, params=parameters)

        data = response.json()['Data']['Data']

        return data


    def data_to_dataframe(data):
        # data from json is in array of dictionaries
        df = pd.DataFrame.from_dict(data)
        # time is stored as an epoch, we need normal dates
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df


    def create_crypto_data(close_price_df):

        close_price = close_price_df
        # The indicators we'll need
        bollinger_up, bollinger_down, sma = get_bollinger_bands(close_price, rate=30)
        bollinger_up = bollinger_up.to_numpy().T
        bollinger_down = bollinger_down.to_numpy().T
        bollinger_difference = bollinger_up - bollinger_down
        sma = sma.to_numpy().T

        MACD_line, ema_of_MACD = MACD(close_price)
        MACD_line = MACD_line.to_numpy().T

        Stochastic_line, K_line = Stochastic(close_price,lim)
        Stochastic_line = np.array(Stochastic_line)
        K_line = np.array(K_line)
        ema100 = close_price.ewm(span=200, adjust=False).mean()
        ema100 = ema100.to_numpy().T
        close_price_dataframe_type = close_price
        ema_of_MACD = ema_of_MACD.to_numpy().T
        close_price = close_price.to_numpy().T
        return  close_price, MACD_line, ema100, ema_of_MACD, close_price_df , Stochastic_line, sma, K_line , bollinger_down,bollinger_up

    percentage_list=[]
    num_of_transactions_list=[]
    wallet_euro_list=[]
    end_minute=100
    rsi_bound=30
    lim=24
    # for lim in range(10,30,2):

    # WE CAN SEARCH WHATEVER CRYPTO PAIR WE LIKE
    cryptos = ['BTC', 'ETH', 'ADA','BNB', 'XRP', 'LUNA', 'SOL','AVAX','DOT', 'SHIB', 'MATIC', 'DOGE', 'CRO','ATOM','LTC' \
    , 'TRX', 'LINK','NEAR','UNI','FTT','LEO','BCH','ALGO','XLM']
    cryptos = ['tusd','usdt','busd','ust','dai','usdc','usdp']
    cryptos=['dai']
    point_list = []
    point_list2 = []
    loser_trades = []
    winner_trades = []
    RSI_list = []
    # Implement strategy here and test it with the above data
    
    open_trade = False
    buy_price = 0

    close_price_list = np.zeros((len(close_price_df)))
    MACD_line_list = np.zeros((len(close_price_df)))
    ema100_list = np.zeros((len(close_price_df)))
    sma_list = np.zeros((len(close_price_df)))
    ema_of_MACD_list = np.zeros((len(close_price_df)))
    bollinger_up_list = np.zeros((len(close_price_df)))
    bollinger_down_list = np.zeros((len(close_price_df)))
    Stochastic_line = np.zeros((len(close_price_df)))
    K_line_list = np.zeros((len(close_price_df)))


    i=0
    num_of_transactions = 0
    time_bound2 = time()

    target_currency = 'EUR'

    close_price_list[:], MACD_line_list[:], ema100_list[:], ema_of_MACD_list[:], df, \
    Stochastic_line[:], sma_list[:], K_line_list[:] , bollinger_down_list[:],bollinger_up_list[:] = create_crypto_data(close_price_df)



    def detect_downtrend(prices):
        """
        Detects if there is a downtrend in the given array of BTC prices.

        Parameters:
        prices (numpy.ndarray): Array of BTC prices.

        Returns:
        bool: True if downtrend is detected, False otherwise.
        """
        if len(prices) < 2:
            raise ValueError("Array must contain at least two prices.")

        # Calculate the difference between consecutive prices
        price_diff = np.diff(prices)

        # Check if all differences are negative, indicating a downtrend
        downtrend = all(diff < 0 for diff in price_diff)

        return downtrend



    for i in range(len(close_price_df)-30,len(close_price_df)):
        #price = float(auth_client.get_product_ticker(product_id='USDT-EUR')['price'])
        # close_price_list = np.delete(close_price_list,0)

        #close_price_list = np.append(close_price_list,price)
        # print('price',price)
        #
        # MACD_line_list[:], ema100_list[:], ema_of_MACD_list[:], close_price_df, \
        # Stochastic_line[:], sma_list[:], K_line_list[:], bollinger_down_list[:] = update_crypto_data(close_price_list)
        #
        #
        #
        # print/(i)
        lastx_list = close_price_list[i - 10:i]  # 140

        is_dowentrend = detect_downtrend(lastx_list)

        
        #print(lastx_list)

        day_min = min(lastx_list)
        rsi_current = rsi(close_price_df.iloc[i- 14:i])
        # rsi_previous = rsi(close_price_df.iloc[-1 - 15:-2])
        # price_list.append(price)

        #
        # buy_price_initial = 0
        #
        #
        # if open_trade:


            # if ((close_price_list[i] < buy_price - stop_limit) and open_trade) or (
            #         (close_price_list[i] > buy_price + stop_limit) and open_trade):
        

            
        # else:
        #

        #
        for minute_before in range(2, 5):
            if (Stochastic_line[i - minute_before - 1] > K_line_list[i - minute_before - 1]) and (
                    Stochastic_line[i - minute_before] < K_line_list[i - minute_before]):
                cross2 = True
                break
            else:
                cross2 = False
            
            if ((ema_of_MACD_list[i] < MACD_line_list[i]) and (
                    ema_of_MACD_list[i-1] > MACD_line_list[i-1])):
                cross3 = True
                break
            
            else:
                cross3 = False

            if (Stochastic_line[i - minute_before] < 15):
                cross = True
                break
            else:
                cross = False
        #
        array_data = [MACD_line_list[i], ema100_list[i], ema_of_MACD_list[i], Stochastic_line[i], sma_list[i],
                    rsi_current]
        array_data = np.array(array_data)
        array_data = array_data.reshape(1, -1)

        #print(loaded_model.predict(array_data))

        if close_price_list[i]>bollinger_up_list[i] or (close_price_list[i]< buy_price - (bollinger_up_list[i]-bollinger_down_list[i])/1 ):
           
        
            point_list.append(i)

            print("YOU HAVE TO SELL:",crypto)

            plt.plot(range(0, len(close_price_df)), close_price_list, label='Closing Prices')
            plt.plot(range(0, len(close_price_df)), bollinger_down_list, label='Bollinger Up', c='black')
            plt.plot(range(0,len(close_price_df)), bollinger_up_list, label='Bollinger Down', c='black')
            plt.plot(range(0, len(close_price_df)), sma_list, label='SMA', c='b')

            for i in range(len(point_list)):
                plt.plot(point_list[i], close_price_list[point_list[i]], 'o',color='red')

            plt.show()

        if (rsi_current < 30) and (not open_trade) and (cross2)  and cross and (

        bollinger_down_list[i] > close_price_list[i]):# and not is_dowentrend:
            # and loaded_model.predict(array_data):
    #       
            print("YOU HAVE TO BUY:",crypto)

            point_list2.append(i)


            print(point_list2)


            plt.plot(range(0, len(close_price_df)), close_price_list, label='Closing Prices')
            plt.plot(range(0, len(close_price_df)), bollinger_down_list, label='Bollinger Up', c='black')
            plt.plot(range(0,len(close_price_df)), bollinger_up_list, label='Bollinger Down', c='black')
            plt.plot(range(0, len(close_price_df)), sma_list, label='SMA', c='b')

            for i in range(len(point_list2)):
                plt.plot(point_list2[i], close_price_list[point_list2[i]],'o',color='green')
            
            plt.show()


            # wallet_btc, wallet_euro = buy(close_price_list[i], wallet_euro, wallet_btc)
            # auth_client.place_market_order(product_id='USDT-EUR',
            #                                    side='buy',
            #                                    funds='20')

            # accounts = auth_client.get_accounts()


            # open_trade = True
            # point_list2.append(i)
            # buy_price = close_price_list[i]
            # buy_price_initial = buy_price

            # buy_price = close_price_list[i]
            # buy_price_initial = buy_price
            # stop_limit = buy_price - day_min
            # day_min2 = day_min
            # num_of_transactions += 1



    # if open_trade:
    #     wallet_btc, wallet_euro = sell(close_price_list[i], wallet_euro, wallet_btc)


#     print('eur', wallet_euro)
#     print('btc', wallet_btc)
#     print('num_of_transactions', num_of_transactions)
#     print('winner_trades', len(winner_trades))
#     print('loser_trades', len(loser_trades))


#     total_transactions += num_of_transactions
#     total_wins += len(winner_trades)
#     total_loss += len(loser_trades)


# print('total_euro', wallet_euro)
# print('total_transactions', total_transactions)
# print('total_wins', total_wins)
# print('total_loss', total_loss)
# print('total_euro_deposit', total_euro_deposit)
# print('total_euro_gain', total_euro_gain)
# print('total_euro_loss', total_euro_loss)

   
# plt.plot(range(0, len(close_price_df)), close_price_list, label='Closing Prices')
# plt.plot(range(0, len(close_price_df)), bollinger_down_list, label='Bollinger Up', c='black')
# plt.plot(range(0,len(close_price_df)), bollinger_up_list, label='Bollinger Down', c='black')
# plt.plot(range(0, len(close_price_df)), sma_list, label='SMA', c='b')

# print(point_list)
# for i in range(len(point_list)):
#     plt.plot(point_list2[i], close_price_list[point_list2[i]],'o',color='green')
# for i in range(len(point_list)):
#     plt.plot(point_list[i], close_price_list[point_list[i]], 'o',color='red')

# plt.show()

