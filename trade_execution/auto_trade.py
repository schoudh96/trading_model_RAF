# -*- coding: utf-8 -*-
"""
Created on Sun May 15 23:51:23 2022

@author: schoudh
"""

"""
1. connect to kite app
2. at open, buy the selected orders from the dataframe 
3. 

"""
#test
#!python
import logging
from kiteconnect import KiteConnect
from kiteconnect import KiteTicker
import pandas as pd
import sqlite3
import numpy as np
import os
import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
import time, pyotp
import math
from datetime import date

def login_in_zerodha(api_key, api_secret, user_id, user_pwd, totp_key):
    driver = uc.Chrome()
    driver.get(f'https://kite.trade/connect/login?api_key={api_key}&v=3')
    login_id = WebDriverWait(driver, 10).until(lambda x: x.find_element_by_xpath('//*[@id="userid"]'))
    login_id.send_keys(user_id)

    pwd = WebDriverWait(driver, 10).until(lambda x: x.find_element_by_xpath('//*[@id="password"]'))
    pwd.send_keys(user_pwd)

    submit = WebDriverWait(driver, 10).until(lambda x: x.find_element_by_xpath('//*[@id="container"]/div/div/div[2]/form/div[4]/button'))
    submit.click()

    time.sleep(1)
    #adjustment to code to include totp
    totp = WebDriverWait(driver, 10).until(lambda x: x.find_element_by_xpath('//*[@id="totp"]'))
    authkey = pyotp.TOTP(totp_key)
    totp.send_keys(authkey.now())
    #adjustment complete

    continue_btn = WebDriverWait(driver, 10).until(lambda x: x.find_element_by_xpath('//*[@id="container"]/div/div/div[2]/form/div[3]/button'))
    continue_btn.click()

    time.sleep(5)

    url = driver.current_url
    initial_token = url.split('request_token=')[1]
    request_token = initial_token.split('&')[0]

    driver.close()

    kite = KiteConnect(api_key = api_key)
    #print(request_token)
    data = kite.generate_session(request_token, api_secret=api_secret)
    kite.set_access_token(data['access_token'])

    return kite

def order_filename(partial_string = 'current_order', dir_p = 'C:/Users/schoudh/Documents/Modeling/Data/'):
    list_files = os.listdir(dir_p)    
    match = list(filter(lambda x:x.startswith(partial_string), list_files))[0]
    date = match.split(['_','.'])[-1]

def placeSLOrder(symbol, buy_sell, quantity, sl, price, target):
    print("IN SL ORDER FUNCTION")

    if buy_sell == "buy":
        t_type=kite.TRANSACTION_TYPE_BUY
    elif buy_sell == "sell":
        t_type=kite.TRANSACTION_TYPE_SELL  
    print(symbol, buy_sell, quantity, price)
    sl_order = kite.place_order(tradingsymbol=symbol,
                    exchange=kite.EXCHANGE_NSE,
                    transaction_type=t_type,
                    quantity=quantity,
                    order_type=kite.ORDER_TYPE_SLM,
                    product=kite.PRODUCT_MIS,
                    trigger_price = price,
                    variety=kite.VARIETY_REGULAR)
    order_id = sl_order
    return order_id

def cancel_order(session, variety, order_id, symbol=None, order_type=None, transaction_type=None):
    """

    Parameters
    ----------
    session : send kiteobj(or kite)
    variety : Regular or AMO or BO or CO
    order_id : order id of the order to cancel
    symbol : tradingsymbol of stock to sell
    order_type : sl-m or limit, etc.
    transaction_type : buy or sell
    
    Returns
    -------
    Success note.

    """
    
    session.cancel_order(variety, order_id)
    print(f"Successfully canceled {transaction_type} type {order_type} order for {symbol} at {pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')}.")
    
def modify_order(session, variety, order_id, quanty= None, price = None, order_type = None, trigger_price = None, symbol=None):
    """

    Parameters
    ----------
    session : send kiteobj(or kite)
    variety : Regular or AMO or BO or CO
    order_id : order id of the order to cancel
    quantity : quantity to change
    price : new price
    order_type: new order product type like sl, slm, limit, etc.
    trigger_price : new trigger price
    symbol : trading symbol of stock to sell
    
    Returns
    -------
    Success note.

    """
    
    session.modify_order(variety, order_id, quanty, price , order_type , trigger_price)
    return None

def instrumentLookup(instrument_df,symbol):
    """Looks up instrument token for a given script from instrument dump"""
    try:
        return instrument_df[instrument_df.tradingsymbol==symbol].instrument_token.values[0]
    except:
        return -1

logging.basicConfig(level=logging.DEBUG)

kiteobj = login_in_zerodha(api_key = "***", api_secret = "***", user_id = '***', user_pwd = '***', totp_key = '***')
print(kiteobj.profile())

kite = kiteobj

#get dump of all NSE instruments
instrument_dump = kite.instruments("NSE")
instrument_df = pd.DataFrame(instrument_dump)


orders = pd.read_csv(open(r'C:\Users\schoudh\Documents\Modeling\Krauss RAF Momentum Model\order_file.txt', 'r').read())
orders = orders.merge(instrument_df[['tradingsymbol', 'instrument_token','name', 'tick_size']], left_on = 'tradingsymbol', right_on = 'tradingsymbol', how = 'inner')

secs = orders['tradingsymbol'].tolist()
secs = ['NSE:' + sec for sec in secs]
orders.loc[orders.close < 2200, 'flag'] = 1
orders = orders[orders['flag'] == 1].head(10)

cash = 1500
margin = pd.DataFrame(kite.margins(segment= 'equity'))['net'].mode().iloc[0]

amount_trade = math.floor(margin - 1500)
orders['port_value'] = amount_trade
orders['security_value'] = orders['port_value']/len(orders)
orders = orders.sort_values('close')
orders['cum_close'] = orders['close'].cumsum()
counter = 0

for i in range(10):
    if i == 0:
        orders['quantity'] = np.round(orders['security_value']/orders['close'])
        expected_portvalue = sum(orders['quantity']*orders['close'])
        rem_cash = amount_trade - expected_portvalue
        print('Counter ', counter,'\nRemaining cash', amount_trade - expected_portvalue)
    else:
        if (orders['cum_close'] < rem_cash).any():
            orders.loc[orders['cum_close'] < rem_cash,'quantity'] = orders.loc[orders['cum_close'] < rem_cash,'quantity'] + 1
            expected_portvalue = sum(orders['quantity']*orders['close'])
            rem_cash = amount_trade - expected_portvalue
            print('Counter ', counter,'\nRemaining cash', amount_trade - expected_portvalue)
        else:
            print('Optimum allocation reached.')
            break
print('Expected portfolio value: ', expected_portvalue)

quote = pd.DataFrame(kite.quote(secs))
quote.columns = quote.columns.str[4:]

order_date = (pd.to_datetime(orders['date'].iloc[0]) + pd.tseries.offsets.BDay(1)).strftime('%Y-%m-%d')

#fail safe 
if not quote.loc['last_trade_time'].iloc[0].strftime('%Y-%m-%d') == order_date:
    raise Exception('Order Date does not match with trade date. Check.')
else:
    print('Date as of trading day.')

quote = quote.loc[['last_price']]
quote = quote.stack()
quote.name = 'Curr_price'
quote = quote.reset_index()
quote.drop(columns = 'level_0', inplace = True)
quote.rename(inplace = True, columns = {'level_1' : 'tradingsymbol'})

orders = orders.merge(quote, on = 'tradingsymbol')
orders['Curr_price'] = orders['Curr_price'].astype(float)

sl = 0.03
orders['stoploss'] = np.round(orders['Curr_price']*sl,1)

# =============================================================================
# if 0.95*margin > (np.sum(orders['quantity']*orders['close'])):
#     print('Success. Portfolio value at/below 90% of margin. Proceed with order.')
# else:
#     raise Exception('Warning! Portfolio Value above 90% margin.')
# 
# =============================================================================
order_book = pd.DataFrame(kite.orders())

holdings = pd.DataFrame(kite.holdings())
holdings = holdings.loc[(holdings['quantity'] != 0) | (holdings['t1_quantity'] != 0)]
#holdings = holdings.loc[(holdings['quantity'] != 0)]
                        
curr_holdings_l = holdings['tradingsymbol'].tolist()

try:
    order_holdings = orders['tradingsymbol'].tolist()
except:
    order_holdings = [orders['tradingsymbol']]
    

#order_holdings
for sec in order_holdings:
    if sec in curr_holdings_l:
        print('Security in Current Holdings. Skipping.')
        pass
    else:
        q = int(np.ceil(orders.loc[orders.tradingsymbol == sec, 'quantity'].iloc[0]))
        
        print(f'Placing order for {sec}.')
        kite.place_order(variety = kite.VARIETY_REGULAR, exchange = kite.EXCHANGE_NSE, tradingsymbol = sec,\
                         transaction_type = kite.TRANSACTION_TYPE_BUY, quantity = q, product = kite.PRODUCT_CNC,\
                         order_type=kite.ORDER_TYPE_MARKET, tag = 'strat1')
 
#place slm order
for sec in order_holdings:
    if sec in curr_holdings_l:
        print(f'Sec {sec} already in current holdings')
        pass
    elif len((order_book.loc[(order_book.tradingsymbol == sec)& (order_book.transaction_type == 'SELL') & (order_book['status'] != 'REJECTED'), 'tradingsymbol'])) > 0:
        print(f'Sec {sec} already in order book')
        pass
        
#    elif (sec in order_book['tradingsymbol'].tolist()) & (order_book.loc[order_book['tradingsymbol'] == sec, 'transaction_type'] == 'SELL').any() & (order_book.loc[order_book['tradingsymbol'] == sec, 'status'] != 'REJECTED').any():
        
    else:
        q = int(np.ceil(order_book.loc[(order_book.tradingsymbol == sec) & (order_book.transaction_type == 'BUY') & (order_book['status'] == 'COMPLETE'), 'filled_quantity'].iloc[0]))
        #sl_amt = orders.loc[orders.tradingsymbol == sec, 'stoploss'].iloc[0]
        price = np.round(order_book.loc[(order_book.tradingsymbol == sec) & (order_book.transaction_type == 'BUY') & (order_book['status'] == 'COMPLETE'), 'average_price'].iloc[0]*(1-sl),1)

        print(f'Placing sl-m order for {q} shares of {sec} at {price}.')
        
        kite.place_order(tradingsymbol=sec,\
                exchange=kite.EXCHANGE_NSE,\
                transaction_type=kite.TRANSACTION_TYPE_SELL ,\
                quantity=q,\
                order_type=kite.ORDER_TYPE_SLM,\
                product=kite.PRODUCT_CNC,\
                trigger_price = price,\
                variety=kite.VARIETY_REGULAR)
        

print('SUCCESS!! Placed orders!')