import pandas as pd
import matplotlib.pyplot as plt
import requests
from time import time
from datetime import datetime
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler
percent=[]

#load the random forest classification model
filename = 'RF_crypto_model_smart.sav'
loaded_model = joblib.load(filename)



pd.set_option('expand_frame_repr', True)

K_list = [0,0,0]
data = pd.read_csv('btc_data.csv', low_memory=False)
#data= data.loc[0,:]
data=data.iloc[::-1]
#data= data.loc[4000:-1]
print(data.head())
close_price_df = data['close']
time_data = data['date']
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
    if av_loss is not 0:
        RS=av_gain/av_loss

    RSI=100-100/(1+RS)

    return RSI


def get_sma(prices, rate):
    return prices.rolling(rate).mean()


def buy(price, wallet_euro, wallet_btc):
    wallet_btc = 1 * 1100 / price
    wallet_euro = wallet_euro - 1 * 1100 - 0.001*1100
    return wallet_btc, wallet_euro


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
    wallet_euro = wallet_euro + wallet_btc * price -1100*0.001
    wallet_btc = 0
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
wallet_euro =1100
wallet_btc = 0
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
# print(Stochastic_line)
# plt.plot(np.arange(0,len(Stochastic_line)),Stochastic_line[:])
# plt.plot(np.arange(0,len(Stochastic_line)),K_line_list[:])
# plt.show()
#
#WE START THE SIMULATION BACKTESTING
# duration_list=[]
#
# for i in range(200, len(close_price_df)):
#     #last14 = close_price_dataframe_type.iloc[i-14:i+1]
#     #last14sma=sma2.iloc[i-10:i]
#     #RSI = rsi(last14)
#     lastx_list=close_price_list[i-end_minute:i+1]      #140
#     rsi_current=rsi(close_price_df.iloc[i-14:i+1])
#     rsi_previous=rsi(close_price_df.iloc[i-15:i])
#
#     #RSI_list.append(RSI)
#     #average_bollinger = sum(bollinger_difference[i-40:i+1])/len(bollinger_up[i-40:i+1])
#
#     # if is_downtrade(last14sma):
#     #     point_list.append(i)
#     # else:
#     #     point_list2.append(i)
#
#     buy_price_initial = 0
#
#     # if bollinger_difference[i]>average_bollinger+0.1*average_bollinger and RSI < 40 and (not open_trade) and (not is_downtrade(last14sma)):
#     #     point_list2.append(i)
#     #     wallet_btc, wallet_euro = buy(close_price[i], wallet_euro, wallet_btc)
#     #     open_trade = True
#     #     buy_price = close_price[i]
#     # if (close_price[i]>buy_price+buy_price*0.003 and open_trade) or (close_price[i] < buy_price-0.001*buy_price and open_trade):
#     #     wallet_btc, wallet_euro = sell(close_price[i], wallet_euro, wallet_btc)
#     #     open_trade = False
#     #     point_list.append(i)
#     if open_trade:
#         # if open_trade and close_price_list[i,crypto] > buy_price:
#         #     buy_price=close_price_list[i,crypto]
#         day_min = min(lastx_list[:])
#
#
#
#
#         # if ((close_price_list[i] < buy_price-stop_limit) and open_trade) or ((close_price_list[i] > buy_price+stop_limit) and open_trade):
#         # #if (close_price[i]>buy_price+buy_price*0.007 and open_trade) or (close_price[i] < buy_price-0.001*buy_price and open_trade):
#         #     wallet_btc, wallet_euro = sell(close_price_list[i], wallet_euro, wallet_btc)
#         if close_price_list[i]>sma_list[i]:
#             open_trade = False
#             if buy_price > close_price_list[i]:
#                 loser_trades.append('oups')
#             else:
#                 winner_trades.append('yesss')
#             point_list.append(i)
#             wallet_btc, wallet_euro = sell(close_price_list[i], wallet_euro, wallet_btc)
#             print('euros', wallet_euro)
#             # print('percentage', len(winner_trades) / num_of_transactions)
#             # print('num_of_transactions',num_of_transactions)
#             percent.append( len(winner_trades) / num_of_transactions)
#             # duration_list.append(i-buy_minute)
#
#
#
#
#
#     else:
#
#         day_min = min(lastx_list[:])
#         # for minute_before in range(2,end_minute):   #5
#         #     #if (Stochastic_line[i-minute_before-1] > K_line_list[i-minute_before-1]) and (Stochastic_line[i-minute_before] < K_line_list[i-minute_before]):
#         #     if  (Stochastic_line[i-minute_before] < 20):
#         #         cross = True
#         #         break
#         #     else:
#         #         cross = False
#         for minute_before in range(2, 5):
#             if (Stochastic_line[i-minute_before-1] > K_line_list[i-minute_before-1]) and (Stochastic_line[i-minute_before] < K_line_list[i-minute_before]):
#                 cross2 = True
#                 break
#             else:
#                 cross2 = False
#
#             if ((ema_of_MACD_list[i] < MACD_line_list[i]) and (
#                          ema_of_MACD_list[i - 1] > MACD_line_list[i - 1])):
#                 cross3 = True
#                 break
#
#             else:
#                 cross3 = False
#             if  (Stochastic_line[i-minute_before] < 20):
#                 cross = True
#                 break
#             else:
#                 cross = False
#         #if (bollinger_down_list[i-1]>close_price_list[i-1]) and (bollinger_down_list[i]<close_price_list[i]) : \
#         # if (rsi_current<20) and (not open_trade) and ((ema_of_MACD_list[i] < MACD_line_list[i]) \
#         # and ema_of_MACD_list[i - 1] > MACD_line_list[i - 1]) and cross and cross2  \
#         #          and   (close_price_list[i] < sma_list[i]):
#         #STRATEGY 1
#         array_data=[MACD_line_list[i],ema100_list[i],ema_of_MACD_list[i],Stochastic_line[i],sma_list[i],rsi_current]
#         array_data=np.array(array_data)
#         array_data=array_data.reshape(1, -1)
#         array_sum = np.sum(array_data)
#         array_has_nan = np.isnan(array_sum)
#         if array_has_nan:
#             continue
#
#         #print((bollinger_down_list[i]>close_price_list[i]))
#         # if (rsi_previous < (rsi_bound))  and (not open_trade) and (cross2 or cross3 or cross ) and (     #CROSS2 (ema100_list[i] < close_price_list[i]) \
#         #      bollinger_down_list[i]>close_price_list[i])  \
#         #    and loaded_model.predict(array_data):
#         if (bollinger_down_list[i]>close_price_list[i]):
#             buy_minute=i
#             #MACD,ema100,ema_MACD,stochastic,sma,rsi
#
#
#                 # and ema_of_MACD_list[i - 1] > MACD_line_list[i - 1]) and cross and cross2  \
#                 #          and   (close_price_list[i] < sma_list[i]):
#
#         #((ema_of_MACD_list[i] < MACD_line_list[i]) and (
#         # if  (ema_of_MACD_list[i ] < 0.7*MACD_line_list[i])  and (rsi_current<40) \
#         #             and (not open_trade) and (bollinger_down_list[i-1]>close_price_list[i-1])\
#         #              and (bollinger_down_list[i]<close_price_list[i]): #and (cross is True):# and (cross2 is False):
#             #check also the rsi value
#             #if RSI < 100 and (not open_trade):
#             point_list2.append(i)
#             wallet_btc, wallet_euro = buy(close_price_list[i], wallet_euro, wallet_btc)
#             open_trade = True
#             buy_price = close_price_list[i]
#             buy_price_initial = buy_price
#             stop_limit = buy_price - day_min
#             day_min2 = day_min
#             num_of_transactions += 1







for i in range(500, len(close_price_df)):
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
    lastx_list = close_price_list[i - 450:i]  # 140
    #print(lastx_list)

    day_min = min(lastx_list)
    rsi_current = rsi(close_price_df.iloc[i- 14:i])
    # rsi_previous = rsi(close_price_df.iloc[-1 - 15:-2])
    # price_list.append(price)

    #
    # buy_price_initial = 0
    #
    #
    if open_trade:


        if ((close_price_list[i] < buy_price - stop_limit) and open_trade) or (
                (close_price_list[i] > buy_price + stop_limit) and open_trade):
        # if close_price_list[i]>bollinger_up_list[i]:
            wallet_btc, wallet_euro = sell(close_price_list[i], wallet_euro, wallet_btc)
            # auth_client.place_market_order(product_id= 'USDT-EUR',
            #                                side='sell',
            #                                size=str(wallet_crypto))
            open_trade = False
            if (close_price_list[i]-buy_price)/buy_price<0.002:
                loser_trades.append('oups')
            else:
                winner_trades.append('yesss')
            point_list.append(i)
            #print('euros', wallet_euro)
            #print((close_price_list[i]-buy_price)/buy_price)
            # print('percentage', len(winner_trades) / num_of_transactions)
            # print('num_of_transactions',num_of_transactions)
            percent.append(len(winner_trades) / num_of_transactions)
    else:
    #

    #
        for minute_before in range(2, 5):
            if (Stochastic_line[i - minute_before - 1] > K_line_list[i - minute_before - 1]) and (
                    Stochastic_line[i - minute_before] < K_line_list[i - minute_before]):
                cross2 = True
                break
            else:
                cross2 = False
            #
            # if ((ema_of_MACD_list[i] < MACD_line_list[i]) and (
            #         ema_of_MACD_list[i-1] > MACD_line_list[i-1])):
            #     cross3 = True
            #     break
            #
            # else:
            #     cross3 = False

            # if (Stochastic_line[i - minute_before] < 15):
            #     cross = True
            #     break
            # else:
            #     cross = False
        #
        array_data = [MACD_line_list[i], ema100_list[i], ema_of_MACD_list[i], Stochastic_line[i], sma_list[i],
                      rsi_current]
        array_data = np.array(array_data)
        array_data = array_data.reshape(1, -1)

        #print(loaded_model.predict(array_data))
        if (rsi_current < 30) and (not open_trade) and (cross2) and (

         bollinger_down_list[i] > close_price_list[i]) \
            and loaded_model.predict(array_data):
    #
            wallet_btc, wallet_euro = buy(close_price_list[i], wallet_euro, wallet_btc)
            # auth_client.place_market_order(product_id='USDT-EUR',
            #                                    side='buy',
            #                                    funds='20')

            # accounts = auth_client.get_accounts()


            open_trade = True
            print('I bought crypto')
            point_list2.append(i)
            buy_price = close_price_list[i]
            buy_price_initial = buy_price

            buy_price = close_price_list[i]
            buy_price_initial = buy_price
            stop_limit = buy_price - day_min
            # day_min2 = day_min
            num_of_transactions += 1



print('eur', wallet_euro)
print('btc', wallet_btc)
print('num_of_transactions', num_of_transactions)
print('winner_trades', len(winner_trades))
print('loser_trades', len(loser_trades))
# #print('percentage', len(winner_trades)/num_of_transactions)
plt.plot(range(0, len(close_price_df)), close_price_list, label='Closing Prices')
plt.plot(range(0, len(close_price_df)), bollinger_down_list, label='Bollinger Up', c='black')
plt.plot(range(0,len(close_price_df)), bollinger_up_list, label='Bollinger Down', c='black')
plt.plot(range(0, len(close_price_df)), sma_list, label='SMA', c='b')

print(point_list)
for i in range(len(point_list)):
    plt.plot(point_list2[i], close_price_list[point_list2[i]],'o',color='green')
for i in range(len(point_list)):
    plt.plot(point_list[i], close_price_list[point_list[i]], 'o',color='red')

plt.show()















# #print(point_list)
# plt.title('Bollinger Bands')
# plt.xlabel('Minutes')
# plt.ylabel('Closing Prices')
# time = df.index.to_numpy()
#
# time=range(0,len(close_price_list))
# time=time_data.to_numpy().T
# plt.plot(time, close_price_list, label='Closing Prices')
# plt.plot(time, bollinger_up, label='Bollinger Up', c='black')
# plt.plot(time, bollinger_down, label='Bollinger Down', c='black')
# plt.plot(time, sma, label='SMA', c='b')
# sma2 = sma2.to_numpy()
#
# #plt.plot(time, sma2, label='SMA', c='yellow')
#
# for i in range(len(point_list)):
#     plt.plot(time[point_list[i]], close_price_list[point_list[i]],'o',color='red')
# for i in range(len(point_list2)):
#     plt.plot(time[point_list2[i]], close_price_list[point_list2[i]], 'o',color='green')
#
# plt.show()
# #print(RSI_list)
# # plt.figure(2)
# # plt.plot(time[180:len(time)],RSI_list)
# #
# # plt.figure(3)
# #
# # plt.plot(time, bollinger_difference, label='Bollinger Difference', c='black')
# plt.figure(2)
# plt.plot(time[180:len(time)],MACD_line[180:len(time)], c='black')
# plt.plot(time[180:len(time)],ema_of_MACD[180:len(time)], c='red')


# print('eur', wallet_euro)
# print('btc', wallet_btc)
# print('num_of_transactions', num_of_transactions)
# print('winner_trades', len(winner_trades))
# print('loser_trades', len(loser_trades))
# print('percentage', len(winner_trades)/num_of_transactions)
# # plt.figure(1)
# plt.plot(range(10,30,2),num_of_transactions_list)
# plt.figure(2)
# plt.plot(range(10,30,2), percentage_list)

