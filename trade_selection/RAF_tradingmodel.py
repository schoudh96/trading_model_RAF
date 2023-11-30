# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 02:07:06 2021

@author: schoudh
"""

""" Functions changed -> 1. get_target_variable
                         2. get_calendar_dates
                         3. create_subuniverse
    from CART code    
"""
import os
import joblib
#set working directory path
path = r"C:\Users\schoudh\Documents\Modeling\Krauss RAF Momentum Model"
os.chdir(path)

if os.getcwd() == path:
    print('cwd changed to denoted path')
else:
    raise Exception('Cwd not set to user defined path.')

#import price_download
import pandas as pd
import numpy as np
import time
#from datetime import date
from pandas.tseries.offsets import BDay
import pdb
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import metrics 
from sklearn.tree import export_graphviz  
from IPython.display import Image  
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import mean_squared_error, r2_score
import helper_functions as util
from importlib import reload
from pandas.tseries.offsets import BDay
from ffn import *
import warnings
warnings.filterwarnings('ignore')

def prices_format(prices, start_dt):
    prices = prices.loc[prices['date'] > start_dt]
    prices_count = prices.groupby('date')['close'].count()
    dates_ffill = prices_count.loc[prices_count < 200].index.tolist()
    #print(dates_ffill)
    for date in dates_ffill:
        print('Bfilling for date, ', date)
        prev_bday = (pd.to_datetime(date) - BDay(1)).strftime('%Y-%m-%d')
        prev_prices = prices[prices['date'] == prev_bday].copy()
        prev_prices['date'] = date 
        prices = prices.loc[prices['date'] != date]
        prices = pd.concat([prices, prev_prices])
    return prices


if __name__ == "__main__":
    
    print('Running for trading model.')
    
    holdings = pd.read_csv(r'C:\Users\schoudh\Documents\Modeling\Data\BSE500_wts10May.csv')
    
    #Added clean prices(ffilled) from 2005-01-01 in this file. 
    #Reformat using original file for older dates using prices_format function.
    
    #prices = pd.read_csv(r'C:\Users\schoudh\Documents\Modeling\Data\BSE500_prices_ffill.csv')
    
    prices_record =  r'C:/Users/schoudh/Documents/Modeling/Krauss RAF Momentum Model/prices_max_date.txt'
    max_prices_file = open(prices_record, 'r').read()

    orig_prices = pd.read_csv(max_prices_file)
    prices = prices_format(orig_prices, '2005-01-01')
    
    mapping = pd.read_excel(r'C:/Users/schoudh/Documents/Modeling/Krauss RAF Momentum Model/mapping.xlsx')
    
    train_step = 1000
    test_step = 250
    k = 20 #no of stocks to buy
    holding_period = 1
    min_feature_period = 1 
    
    classification = True #classification tree metrics
    FloatMcapFlag = False #weighting is Equal wt if set to False else FloatMcapWeighting
    
    start_date = '2009-01-01'
    inference = True
    curr_model = r'C:/Users/schoudh/Documents/Modeling/Krauss RAF Momentum Model/final_period_model_1dhpy_20220906.joblib'
    
    ################################################ input_ends here ################################################################################################################
    
    feature_periods = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,40,60,80,100,120,140,160,180,200,220,240]
    
    feature_periods = [x for x in feature_periods if x >= min_feature_period]
    
    #generate the universe 

    #check for membership on that date
    mask = holdings.copy()

    #set date column to name 'Date'
    if not isinstance(mask[mask.columns[0]].iloc[0], (int, float)):
        mask.rename(columns = {mask.columns[0] : 'Date'}, inplace = True)
    else:
        raise ValueError("Check if the first column is the date column.")

    #set for membership in the index
    mask.loc[:,~mask.columns.str.contains('Date')] = mask.loc[:,~mask.columns.str.contains('Date')].notnull().astype(int)


    mask2 = mask.set_index('Date').stack()
    mask2 = mask2[mask2 == 1].reset_index()
    
    holdings = mask2.copy()
    del mask2

    holdings.columns = ['date', 'SQ_Id', 'membership']

    dates = pd.DataFrame()
    eom = pd.Series(pd.date_range(start = max(holdings['date'].min(), prices['date'].min()), end = min(holdings['date'].max(), prices['date'].max()), freq = 'M')).dt.strftime('%Y-%m-%d')
    b_eom = pd.Series(pd.date_range(start = max(holdings['date'].min(), prices['date'].min()), end = min(holdings['date'].max(), prices['date'].max()), freq = 'BM')).dt.strftime('%Y-%m-%d')
    dates['eom'] = eom 
    dates['b_eom'] = b_eom
    
    
    #added this line of 9th May
    holdings['date'] = pd.to_datetime(holdings['date']).dt.strftime('%Y-%m-%d')
    holdings1 = holdings.merge(dates, left_on = 'date', right_on = 'eom', how = 'inner')
    
    if holdings1.shape[0] != holdings.shape[0]:
        raise Exception('Shape of DataFrame has changed post merge. Initial DataFrame might have missing months of data.')
    else:
        print('No missing months of data in DataFrame')
        holdings = holdings1.copy()
        del holdings1
    
    holdings['date'] = holdings['b_eom']
    holdings.drop(columns = ['eom', 'b_eom'], inplace = True)
    
    prices = prices.drop(columns = ['Unnamed: 0'])
    prices.head()
    
    s_holdings = holdings.shape
    c_holdings = holdings.groupby('date').count()
    
    final_holdings_date = holdings['date'].max()
    
    final_month = holdings[holdings.date == final_holdings_date]
    final_month['date'] = prices['date'].max()
    
    holdings = pd.concat([holdings, final_month])
    
    ph = prices.merge(holdings, on = ['date', 'SQ_Id'], how = 'left')
    ph = ph.sort_values(by = ['SQ_Id', 'date'])
    ph['ym'] = pd.to_datetime(ph['date']).dt.strftime('%Y-%m')

    
    ph['membership2'] = ph.groupby(['ym', 'SQ_Id'])['membership'].bfill()
    #ph.loc[ph['ym'].between('2019-12','2020-12')].to_csv('check_ph2.csv')
    univ = ph.dropna(subset=['membership2'])
    #univ.groupby('date').count().to_csv('univ_count2.csv')

    #remove days with only few stocks and prices
    univ.columns
    
    #reload(helper_functions)
    print("Calculating forward n-day return")
    target_variable = util.get_target_variablev5(univ, period = holding_period, selection_top_pct = 0.5) #selection_top_pct signifies cross-sectional return% of a day above which stocks have to give returns to be selected
    
    print("Calculating look back n-day momentum")
    momentum_values = util.get_momentum_value3(target_variable, 'close', feature_periods)
    momentum_col_names = ["Momentum_" + str(freq) + "_days" for freq in feature_periods]    
    
    #momentum_clean_df = util.cleantable(momentum_values, subset = momentum_col_names + ['HpyReturn'])
    #max_date = momentum_values.date.max()
    dates_range = np.unique(momentum_values.date.sort_values()).tolist()
    max_hpyret_date = dates_range[-1-holding_period]
    
    #check
    check1 = momentum_values.loc[momentum_values.date > max_hpyret_date]
    #clean the momentum data
    momentum_clean_df1 = momentum_values.loc[momentum_values.date <= max_hpyret_date].dropna(subset = momentum_col_names + ['HpyReturn'])
    momentum_clean_df2 = momentum_values.loc[momentum_values.date > max_hpyret_date].dropna(subset = momentum_col_names)
    momentum_clean_df = pd.concat([momentum_clean_df1, momentum_clean_df2])
    momentum_clean_df = momentum_clean_df.sort_values(['date','SQ_Id'])
    #momentum_clean_df4 = cleantable(momentum_values, subset = momentum_col_names)
    
    del momentum_clean_df1
    del momentum_clean_df2
    
    momentum_clean_df = momentum_clean_df.loc[momentum_clean_df['date'] >= start_date]
    
    daterange = sorted(np.unique(momentum_clean_df.date).tolist())
    
    ####################################################### model fit and pred ###############################################################################################################################################
    final_trading_port_bkt = pd.DataFrame()
    flag = 0 
    if not inference:
        for i in range(0, len(momentum_clean_df), test_step):
            
            try:
                temp = daterange[i+train_step+test_step]
                train_daterange = daterange[i:i+train_step]
                test_daterange = daterange[i+train_step: i + train_step + test_step:holding_period]
                print(f"Running for final test date : {daterange[i+train_step+test_step]} ")
            except:
                flag = 1
                prev_traindates = train_daterange
                prev_testdates = test_daterange
                train_daterange = train_daterange + daterange[i+train_step-test_step: i + train_step]
                test_daterange = daterange[i+train_step::holding_period]
                
                final_traindates = train_daterange
                final_testdates = test_daterange
                
                if (test_daterange[-1] != daterange[-1]) & (flag == 1):
                    test_daterange.append(daterange[-1])
                
                
                print("Last loop for train and test set...")
                
            
            train_df = momentum_clean_df[momentum_clean_df.date.isin(train_daterange)]
            test_df = momentum_clean_df[momentum_clean_df.date.isin(test_daterange)]
            
            #add models to run here
            #run the random forest model
            
            
            
            print("Fitting the random forest model")
            clf = RandomForestClassifier(max_depth = 20, n_estimators=150, max_features="sqrt", random_state=1)
            clf.fit( train_df.loc[:, train_df.columns[train_df.columns.str.contains('Momentum_')]].values, train_df.loc[:, 'ReturnBucket'].values)
            
            
            
            
            print("\n \n Predicting the top k stocks to buy")
            # predictions and probability of prediction
            predicted_returns = clf.predict(test_df.loc[:, test_df.columns[test_df.columns.str.contains('Momentum_')]].values)
            predicted_returns_prob = clf.predict_proba(test_df.loc[:, test_df.columns[test_df.columns.str.contains('Momentum_')]].values)
            predicted_returns_prob_df = pd.DataFrame(predicted_returns_prob, columns= ['class_0','class_1'])
            predicted_returns_prob_df.index = test_df.index
            
            test_df['return_bucket_predicted'] = predicted_returns
            test_df_ext = test_df.copy()
            test_df_ext['returns_class_0_prob'] = predicted_returns_prob_df['class_0']
            test_df_ext['returns_class_1_prob'] = predicted_returns_prob_df['class_1']
            
            #test_df_ext['port_select'] = 0
            test_df_ext2 = test_df_ext.reset_index(drop = True)
            test_df_ext2.sort_values(by = ['date', 'returns_class_1_prob'], inplace = True, ascending = [True, False])
             
            #test_df_ext2[test_df_ext2.index.isin(test_df_ext2.groupby('Date').head(20).index), 'port_select'] = 1
            
            final_trading_port = test_df_ext2.groupby('date').head(k)
            
            if flag == 1:
                final_trading_port_non_overlap = final_trading_port[~final_trading_port.date.isin(final_trading_port_bkt.date)]
                final_trading_port_bkt = final_trading_port_bkt.append(final_trading_port_non_overlap)
                break
            
            else:
                final_trading_port_bkt = final_trading_port_bkt.append(final_trading_port)
    else:
        
        test_date_start = daterange.index('2022-09-06')
        test_daterange = daterange[test_date_start::holding_period]
        test_df = momentum_clean_df[momentum_clean_df.date.isin(test_daterange)]
        
        clf = joblib.load(curr_model)
        print("\n \n Predicting the top k stocks to buy")
        # predictions and probability of prediction
        predicted_returns = clf.predict(test_df.loc[:, test_df.columns[test_df.columns.str.contains('Momentum_')]].values)
        predicted_returns_prob = clf.predict_proba(test_df.loc[:, test_df.columns[test_df.columns.str.contains('Momentum_')]].values)
        predicted_returns_prob_df = pd.DataFrame(predicted_returns_prob, columns= ['class_0','class_1'])
        predicted_returns_prob_df.index = test_df.index
        
        test_df['return_bucket_predicted'] = predicted_returns
        test_df_ext = test_df.copy()
        test_df_ext['returns_class_0_prob'] = predicted_returns_prob_df['class_0']
        test_df_ext['returns_class_1_prob'] = predicted_returns_prob_df['class_1']
        
        #test_df_ext['port_select'] = 0
        test_df_ext2 = test_df_ext.reset_index(drop = True)
        test_df_ext2.sort_values(by = ['date', 'returns_class_1_prob'], inplace = True, ascending = [True, False])
         
        #test_df_ext2[test_df_ext2.index.isin(test_df_ext2.groupby('Date').head(20).index), 'port_select'] = 1
        
        final_trading_port = test_df_ext2.groupby('date').head(k)
        final_trading_port_bkt = final_trading_port_bkt.append(final_trading_port)
     
    full_trading_port = final_trading_port_bkt.merge(mapping, on = 'SQ_Id', how = 'left') 
    period_trading_port = final_trading_port.merge(mapping, on = 'SQ_Id', how = 'left') 
    
    
    latest_order = period_trading_port[period_trading_port.date == period_trading_port.date.max()]
    latest_order = latest_order[['date','tradingsymbol', 'isin', 'high', 'low', 'close']]
    
    trading_port_file = r'C:\Users\schoudh\Documents\Modeling\Data\current_order_' + (pd.to_datetime(latest_order['date'].iloc[0]) + pd.tseries.offsets.BDay(1)).strftime('%Y%m%d') + '.csv'
    latest_order.to_csv(trading_port_file, index = False)
    
    order_file = open(r'C:\Users\schoudh\Documents\Modeling\Krauss RAF Momentum Model\order_file.txt', 'w')
    order_file.write(trading_port_file)
    order_file.close()
    
    #latest_order
# =============================================================================
#     latest_order['port_value'] = 18000
#     latest_order['security_value'] = latest_order['port_value']/len(latest_order)
#     latest_order['quantity'] = np.round(latest_order['security_value']/latest_order['close'])
#     expected_portvalue = sum(latest_order['quantity']*latest_order['close'])
#     print(expected_portvalue)
# =============================================================================
    
    
    
    
    #Check Feature Importance of latest period
    #########################################################################################################
    start_time = time.time()
    importances = clf.feature_importances_
    std = np.std([
        tree.feature_importances_ for tree in clf.estimators_], axis=0)
    elapsed_time = time.time() - start_time
    
    print(f"Elapsed time to compute the importances: "
          f"{elapsed_time:.3f} seconds")
            
    #feature_names = [f'feature {i}' for i in range(.shape[1])]
    forest_importances = pd.Series(importances, index = momentum_col_names).sort_values()
    
    ##########################################################################################################
    
# =============================================================================
#     AUM = 1e6
# =============================================================================
    
    #Add 1 business date to dates so that portfolio as of next day open
    #final_trading_port_bkt['date'] = final_trading_port_bkt['date'].apply(lambda x: pd.to_datetime(x) + BDay(1)).dt.strftime("%Y-%m-%d")
    
# =============================================================================
#     final_trading_port_bkt['turnover'] = 0
#     counter = 0
#     dates = np.unique(final_trading_port_bkt.date.sort_values()).tolist()
#     total_turnover = 0
#     daily_turnover = 0
#     
#     for i in range(len(dates)):
#         if i == 0:
#             continue
#         else:
#             curr_port = final_trading_port_bkt.loc[final_trading_port_bkt['date'] == dates[i], 'ThirdPartyId']
#             prev_port = final_trading_port_bkt.loc[final_trading_port_bkt['date'] == dates[i-1], 'ThirdPartyId']
#             turnover = len(curr_port[~curr_port.isin(prev_port)])
#             total_turnover += turnover
#             daily_turnover = total_turnover/i
#     
#     print(f"Avg daily turnover: {daily_turnover}")
# =============================================================================
    
# =============================================================================
#     final_trading_port_bkt['weight'] = final_trading_port_bkt.groupby('date')['SQ_Id'].transform(lambda x: 1/x.count())
#     final_trading_port_bkt['Shares'] = (AUM*final_trading_port_bkt['weight'])/final_trading_port_bkt['close']
#     final_trading_port_bkt['AUM'] = AUM
#     finalport = final_trading_port_bkt[['date','SQ_Id', 'close', 'AUM', 'Shares','weight']]  
#     
#     
#     finalport['date'] = pd.to_datetime(finalport['date']) 
#     
#     finalport_wins = final_trading_port_bkt.copy()
#     max_return = 0.35
#     finalport_wins.loc[final_trading_port_bkt['HpyReturn'] > max_return, 'HpyReturn'] = max_return
#     finalport_wins.loc[final_trading_port_bkt['HpyReturn'] < -max_return, 'HpyReturn'] = -max_return
#     finalport_wins['date'] = pd.to_datetime(finalport_wins['date']) 
#     
#     rets = finalport_wins.groupby('date')['HpyReturn'].mean()
#     levels = 100*(1+rets).cumprod()
#     aumlevels = AUM*(1+rets).cumprod()
#         
#     #Backtest
#     
#     perf = levels.calc_stats()
#     perf.plot()
#     
#     perf.display()
#     perf.stats
#     
#     #transaction
#     finalport_wins['finalprice'] = (1+finalport_wins['HpyReturn'])*(finalport_wins['close'])
#     reload(util)
#     
#     turnover_val = util.turnover(finalport_wins[['date','SQ_Id', 'weight']])
#     tot_turn = turnover_val.mean().iloc[0]*2
#     dp_charges = 15*k*tot_turn/2
#     aumlevels = pd.DataFrame(aumlevels)
#     aumlevels['stt'] = tot_turn*aumlevels['HpyReturn']*0.001
#     aumlevels['finallevels'] = aumlevels['HpyReturn'] - aumlevels['stt'] - dp_charges
# =============================================================================
    
    #finalport.rename(inplace = True, columns = {'BMonthEndDates':'Date'})
    #finalport1 = finalport.head(1000)
# =============================================================================
#     im.LoadPortfolio(finalport,PORTFOLIOID,PORTFOLIONAME)
#     
#     im.ResearchIndexLevelsTurnover_McapCorpActions([PORTFOLIOID])
#     #im.ResearchIndexLevelsTurnover([PORTFOLIOID],alternateweighting=False)
#     
#     im.GenerateIndexReports([BENCHMARKID,PORTFOLIOID+'GR'], rolling = True, nonusdvariants = ['INR'], ratios=False,holdings = False, attribution=False, simulationname=SIMULATIONNAME)
#             
# =============================================================================
            
    
        
        