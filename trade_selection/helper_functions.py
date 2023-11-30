show_cu# -*- coding: utf-8 -*-
"""
Created on Fri May 14 20:30:25 2021

@author: schoudh
"""


import pandas as pd
import numpy as np
import time
from datetime import date
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
import getpass

def formatting(df):
    df = df[df.columns[~df.columns.str.contains('_y')].tolist()]
    df.columns = df.columns.str.rstrip('_x')
    return df


def create_sql_temptable(df, table_name = 'StagingData..SC_tempuniverse'):
    """ Currently not to be used """
    sqltable = pd.io.sql.get_schema(df, table_name)
    sqltable = sqltable.replace("\n", '')
    sqltable = sqltable.replace('"', '')
    sqltable = sqltable.replace("TEXT", 'VARCHAR(255)')
    sqltable = sqltable.replace("REAL", 'FLOAT')
    try:
        db.DatabaseDoQuery(sqltable)
    except: 
        print("Table already exists in DB")
    
def insert_sql_temptable(df, table_name = 'StagingData..SC_tempuniverse'):
    """ Currently not to be used """ 
    db.DatabaseDoQuery("Insert Into " + table_name + " values "+str((('?',)*df.columns.size)), df.loc[:,['PortfolioEffectiveDate', 'DataDate', 'PerformanceId', 'CompanyId',
       'ThirdPartyId', 'SectorId', 'SizeSegment', 'TotalOutstandingShares',
       'FloatFactor', 'BMonthEndDates', 'ScdDates']])
    
def returnseries2(ser):
    """ Currently not to be used """
    pdb.set_trace()
    returnser = pd.Series([np.nan]*len(ser))
    for counter in range(len(ser)):
        if counter == 0:
            returnser.iloc[0] = np.nan
        returnser.iloc[counter] = ser.iloc[counter]/ser.iloc[counter - 1]
    ser = returnser
    return ser

def returnseries(ser):
    """ Returns the monthly return series """
    return ser.rolling(window=2).apply(lambda x: (x[1]/x[0]) - 1)
    
def returnseries3(ser, period):
    """ Returns the monthly return series for a given period"""
    return ser.rolling(window=period+1).apply(lambda x: (x[period]/x[0]) - 1)


def returnserbucket(ser):
    """ Returns binary label 1,0 depending on whether stock return is above median monthly return """
    median = ser.median()
    return ser.apply(lambda x : 1 if x>=median else 0)

def returnserbucketv2(ser, quantile_pct = 0.5):
    """ Returns binary label 1,0 depending on whether stock return is above median monthly return """
    quantile_val = ser.quantile(quantile_pct)
    return ser.apply(lambda x : 1 if x>=quantile_val else 0)

def get_target_variable2(table_name, period):
    """ Returns dataframe with outperform and underperform stocks
        Args: 
            table_name : sql_table_name
            period : forward return calculation period
    """
    sqlquery = """ select a.*, b.ThirdPartyId as ThirdPartyIdEDD, b.Date, b.CAP_AdjPrice
    from """ + table_name + """ a
    left join TimeSeries..EquityDailyData b
    on a.ThirdPartyId = b.ThirdPartyId
    and a.ExtendedDates = b.Date 
    order by b.Date DESC"""
    
    ClosePrices = dc.sqlread(sqlquery, 'idnpddb61')
    ReturnSeries = ClosePrices.sort_values(['ThirdPartyIdEDD','Date']).copy()
    ReturnSeries['ReturnSeries'] = ReturnSeries.groupby(['ThirdPartyIdEDD'])['CAP_AdjPrice'].apply(returnseries3, period = period)
    ReturnSeries['ReturnSeries'] = ReturnSeries['ReturnSeries'].shift(periods = -period)
    ReturnSeries['ReturnBucket'] = ReturnSeries.groupby(['Date'])['ReturnSeries'].apply(returnserbucket)
    return ReturnSeries

def STRIcalc(ser):
    initial = 100
    stri = [initial]
    for i in range(len(ser)):
        if i == 0:
            continue
        next = initial*(1 + ser.iloc[i-1])
        stri.append(next)
        initial = next
    stri = pd.Series(stri, index = ser.index)
    return stri
    
def STRIcalc2(ser):
    initial = 100
    stri = []
    for i in range(len(ser)):
        next = initial*(1 + ser.iloc[i])
        stri.append(next)
        initial = next
    stri = pd.Series(stri, index = ser.index)
    return stri

def get_target_variable3(table_name, period):
    """ Returns dataframe with outperform and underperform stocks
        Args: 
            table_name : sql_table_name
            period : forward return calculation period
    """
    sqlquery = """ select a.*, b.ThirdPartyId as ThirdPartyIdEDD, b.Date, b.CAP_AdjPrice, b.[Close], b.LocalClose
    from """ + table_name + """ a
    join TimeSeries..EquityDailyData b
    on a.ThirdPartyId = b.ThirdPartyId
    and a.ExtendedDates = b.Date 
    order by b.Date DESC"""
    
    ClosePrices = dc.sqlread(sqlquery, 'idnpddb61')
    ReturnSeries = ClosePrices.sort_values(['ThirdPartyIdEDD','Date']).copy()
    ReturnSeries['ReturnSeries'] = ReturnSeries['Close']/ReturnSeries['CAP_AdjPrice'] - 1
    #ReturnSeries['ReturnSeries'] = ReturnSeries['ReturnSeries'].shift(periods = -period)
    ReturnSeries['STRI'] = ReturnSeries.sort_values(['ThirdPartyIdEDD','Date']).groupby(['ThirdPartyIdEDD'])['ReturnSeries'].apply(STRIcalc)
    ReturnSeries['HpyReturn'] = ReturnSeries.sort_values(['ThirdPartyIdEDD','Date']).groupby(['ThirdPartyIdEDD'])['STRI'].apply(returnseries3, period = period)
    ReturnSeries['HpyReturn'] = ReturnSeries['HpyReturn'].shift(periods = -period)
    ReturnSeries['ReturnBucket'] = ReturnSeries.groupby(['Date'])['HpyReturn'].apply(returnserbucket)
    return ReturnSeries

def get_target_variablev4(table_name, period, selection_top_pct = 0.5):
    """ Returns dataframe with outperform and underperform stocks
        Args: 
            table_name : sql_table_name
            period : forward return calculation period
    """
    sqlquery = """ select a.*, b.ThirdPartyId as ThirdPartyIdEDD, b.Date, b.CAP_AdjPrice, b.[Close], b.LocalClose
    from """ + table_name + """ a
    join TimeSeries..EquityDailyData b
    on a.ThirdPartyId = b.ThirdPartyId
    and a.ExtendedDates = b.Date 
    order by b.Date DESC"""
    
    ClosePrices = dc.sqlread(sqlquery, 'idnpddb61')
    ReturnSeries = ClosePrices.sort_values(['ThirdPartyIdEDD','Date']).copy()
    ReturnSeries['ReturnSeries'] = ReturnSeries['Close']/ReturnSeries['CAP_AdjPrice'] - 1
    #ReturnSeries['ReturnSeries'] = ReturnSeries['ReturnSeries'].shift(periods = -period)
    ReturnSeries['STRI'] = ReturnSeries.sort_values(['PerformanceId','Date']).groupby(['PerformanceId'])['ReturnSeries'].apply(STRIcalc2)
    ReturnSeries['HpyReturn'] = ReturnSeries.sort_values(['PerformanceId','Date']).groupby(['PerformanceId'])['STRI'].apply(returnseries3, period = period)
    ReturnSeries['HpyReturn'] = ReturnSeries['HpyReturn'].shift(periods = -period)
    ReturnSeries['ReturnBucket'] = ReturnSeries.groupby(['Date'])['HpyReturn'].apply(returnserbucketv2, quantile_pct = selection_top_pct)
    return ReturnSeries

def get_target_variablev5(ReturnSeries, period, selection_top_pct = 0.5):
    """ Returns dataframe with outperform and underperform stocks
        Args: 
            ReturnSeries : sql_table_name
            period : forward return calculation period
    """

    ReturnSeries = ReturnSeries.sort_values(['SQ_Id','date'])
    ReturnSeries['HpyReturn'] = ReturnSeries.groupby('SQ_Id')['close'].pct_change(periods = period).shift(-period)
    ReturnSeries['ReturnBucket'] = ReturnSeries.groupby(['date'])['HpyReturn'].apply(returnserbucketv2, quantile_pct = selection_top_pct)
    return ReturnSeries

def get_target_variable_rds(table_name, period):
    """ Returns dataframe with outperform and underperform stocks
        Args: 
            table_name : sql_table_name
            period : forward return calculation period
    """
    sqlquery = """ select a.*, b.ThirdPartyId as ThirdPartyIdEDD, b.Date, b.CAP_AdjPrice
    from """ + table_name + """ a
    left join TimeSeries..EquityDailyData b
    on a.ThirdPartyId = b.ThirdPartyId
    and a.ExtendedDates = b.Date 
    order by b.Date DESC"""
    
    ClosePrices = dc.sqlread(sqlquery, 'idnpddb61')
    ReturnSeries = ClosePrices.sort_values(['ThirdPartyIdEDD','Date']).copy()
    ReturnSeries['ReturnSeries'] = ReturnSeries.groupby(['ThirdPartyIdEDD'])['CAP_AdjPrice'].apply(returnseries3, period = period)
    ReturnSeries['ReturnSeries'] = ReturnSeries['ReturnSeries'].shift(periods = -period)
    ReturnSeries['ReturnBucket'] = ReturnSeries.groupby(['Date'])['ReturnSeries'].apply(returnserbucket)
    return ReturnSeries
   
def get_target_variable(table_name):
    """ Returns dataframe with outperform and underperform stocks
        Args: 
            table_name : sql_table_name
    """
    sqlquery = """ select a.*, b.ThirdPartyId as ThirdPartyIdEDD, b.Date, b.CAP_AdjPrice
    from """ + table_name + """ a
    left join TimeSeries..EquityDailyData b
    on a.ThirdPartyId = b.ThirdPartyId
    and a.ExtendedDates = b.Date """
    
    ClosePrices = dc.sqlread(sqlquery, 'idnpddb61')
    ReturnSeries = ClosePrices.sort_values(['ThirdPartyIdEDD','Date']).copy()
    ReturnSeries['ReturnSeries'] = ReturnSeries.groupby(['ThirdPartyIdEDD'])['CAP_AdjPrice'].apply(returnseries)
    ReturnSeries['ReturnSeries'] = ReturnSeries['ReturnSeries'].shift(periods = -1)
    ReturnSeries['ReturnBucket'] = ReturnSeries.groupby(['Date'])['ReturnSeries'].apply(returnserbucket)
    return ReturnSeries

def get_inputvars(df, table_name, input_cols):
    """ Returns dataframe with s/p ratio input
        Args: 
            df : input dataframe
            table_name : sql_table_name
            input_cols : column name of input table/ string of column names
    """
    
    sqlquery = """ select a.*, b.ShareClassId, b.ThirdPartyId as ThirdPartyIdInput, b.Date as Date_inputs, """ + input_cols + """
    from """ + table_name + """ a
    left join StagingData..KVR_StyleUniverse b
    on a.ThirdPartyId = b.ThirdPartyId
    and YEAR(a.BMonthEndDates) = YEAR(b.Date) and (MONTH(a.BMonthEndDates) - 1) = MONTH(b.Date) """
    
    inputs = dc.sqlread(sqlquery, 'idnpddb61').sort_values(['ThirdPartyId', 'BMonthEndDates'])
    inputs.loc[:,['ShareClassId',
       'ThirdPartyIdInput', 'Date_inputs', 'EarningsYield', 'CashFlowYield',
       'RevenueYield', 'EarningsGrowth']]  = inputs.\
        groupby('ThirdPartyId')[['ShareClassId',
       'ThirdPartyIdInput', 'Date_inputs', 'EarningsYield', 'CashFlowYield',
       'RevenueYield', 'EarningsGrowth']].ffill().bfill()
        
    merge = df.merge(inputs, on = ['ThirdPartyId', 'BMonthEndDates'], how = 'left')
    total_inputs = formatting(merge)
    
    return total_inputs
    
def get_ROA_input(df, table_name, input_cols):
    """ Returns dataframe with s/p ratio input
        Args: 
            df : input dataframe
            table_name : sql_table_name
            input_cols : column name of input table/ string of column names
    """
    sqlquery = """ select a.*, b.CompanyId as CompanyIdInput, b.Date as Date_ROA, """ + input_cols + """
    from """ + table_name + """ a
    left join Ts..ROA b
    on a.CompanyId = b.CompanyId
    and a.BMonthEndDates = b.Date"""
    
    inputs = dc.sqlread(sqlquery, 'idnpddb61').sort_values(['CompanyId', 'ThirdPartyId', 'BMonthEndDates'])
    inputs.loc[:,['CompanyIdInput', 'Date_ROA', 'ROA']]  = inputs.\
        groupby(['CompanyId', 'ThirdPartyId'])[['CompanyIdInput', 'Date_ROA', 'ROA']].ffill().bfill()
        
    merge = df.merge(inputs, on = ['ThirdPartyId', 'BMonthEndDates'], how = 'left')
    total_inputs = formatting(merge)
    
    return total_inputs

def get_momentum1m_input(df, col_name): #not to be used
    """ Returns 1 month momentum values
        Args: 
            df : input dataframe
            ReturnsColumn(str) : Name of Return Column in df
    """
    df = df.sort_values(['ThirdPartyId', 'BMonthEndDates'])
    df['Momentum1m'] = df.groupby('ThirdPartyId')[col_name].apply(momentumser)
    return df

def momentumser(ser, freq): #functional
    """Returns momentum for n-day period 
       Momentum_n_day = P(t)/P(t-freq) - 1 
        Args:
            ser : ser to calculate momentum over
            freq : period to calculate momentum over
    """
    
    return ser.rolling(freq + 1).apply(lambda x: x[freq]/x[0] - 1)
    

def get_momentum_value(df, target_col, freq): #functional
    """ Returns n-day momentum values
        Momentum_n_day = P(t)/P(t-n) - 1
        
        Args: 
            df : input dataframe
            target_col(str) : column to calculate momentum for
            freq : int or list - of days to calculate momentum for
    """
    
    df = df.sort_values(['ThirdPartyId', 'ExtendedDates'])
    if type(freq) == int:
        freq = list[freq]
    else:
        pass 
    
    for period in freq:
        df['Momentum_'+str(period)+'_days'] = df.groupby('ThirdPartyId')[target_col].apply(momentumser, freq = period)
    
    return df


def get_momentum_value2(df, target_col, freq): #functional
    """ Returns n-day momentum values
        Momentum_n_day = P(t)/P(t-n) - 1
        
        Args: 
            df : input dataframe
            target_col(str) : column to calculate momentum for
            freq : int or list - of days to calculate momentum for
    """
    
    df = df.sort_values(['PerformanceId', 'ExtendedDates'])
    if type(freq) == int:
        freq = list[freq]
    else:
        pass 
    for period in freq:
        df['Momentum_'+str(period)+'_days'] = df.groupby('PerformanceId')[target_col].apply(momentumser, freq = period)
    
    return df

def get_momentum_value3(df, target_col, freq): #functional
    """ Returns n-day momentum values
        Momentum_n_day = P(t)/P(t-n) - 1
        
        Args: 
            df : input dataframe
            target_col(str) : column to calculate momentum for
            freq : int or list - of days to calculate momentum for
    """
    
    df = df.sort_values(['SQ_Id', 'date'])
    if type(freq) == int:
        freq = list((freq,))
    else:
        pass 
    for period in freq:
        print(period)
        
        df['Momentum_'+str(period)+'_days'] = df.groupby('SQ_Id')[target_col].apply(momentumser, freq = period)
    
    return df

def cutoff_date_func(df, column_name, train_pct = 0.7):
    """ Returns the cut-off date for splitting the dataset at the desired split percentage""" 
    ser = df[column_name]
    year = int((ser.max().year - ser.min().year)*train_pct)
    month = 4 
    
    cut_off_date = pd.to_datetime(np.unique(ser[(ser.dt.year == ser.min().year + year) & (ser.dt.month == month)])[0])
    return cut_off_date

def input_transform_quintile(ser):
    """ Returns the data into 5 equal buckets of quantiles numbered 1 to 5 """
    quint1 = ser <= ser.quantile(.2)
    quint2 = (ser <= ser.quantile(.4)) & (ser > ser.quantile(.2)) 
    quint3 = (ser <= ser.quantile(.6)) & (ser > ser.quantile(.4))
    quint4 = (ser <= ser.quantile(.8)) & (ser > ser.quantile(.6))
    quint5 = (ser <= ser.quantile(1)) & (ser > ser.quantile(.8))
    
    ser = ser.mask(quint1, 1).mask(quint2, 2).mask(quint3, 3).mask(quint4, 4).mask(quint5, 5)
    return ser

def input_transform_quintile2(ser):
    """ Returns the data into 5 equal buckets of quantiles numbered 1 to 5 """
    quint1 = ser <= ser.quantile(.33)
    quint2 = (ser <= ser.quantile(.67)) & (ser > ser.quantile(.33)) 
    quint3 = (ser <= ser.quantile(1)) & (ser > ser.quantile(.67))
    
    ser = ser.mask(quint1, 1).mask(quint2, 2).mask(quint3, 3)
    return ser
    
def get_calendar_dates(Mindate, range_freq = 'BM'):
    """ Returns the calendar of the Scd Dates and the monthly BMonthEndDates """
    calendar_DTIndex = pd.date_range(start = Mindate, end = pd.to_datetime('today').strftime('%Y-%m-%d'), freq = range_freq)
    calendar = calendar_DTIndex.strftime('%Y-%m-%d')
    calendar = pd.DataFrame(data = {'ExtendedDates' : calendar})
    calendar['ScdDataDate'] = np.nan
    calendar['ScdDataDate'] = calendar['ExtendedDates'].apply(lambda x: pd.offsets.BMonthEnd().rollforward(x).strftime("%Y-%m-%d") if pd.to_datetime(x).month in [2,5,8,11] else np.nan)
    calendar['ScdDataDate'] = calendar['ScdDataDate'].fillna(method = 'ffill', axis = 0)
    calendar.dropna(how = 'any', inplace = True)
    calendar.ExtendedDates = pd.to_datetime(calendar.ExtendedDates)
    calendar.ScdDataDate = pd.to_datetime(calendar.ScdDataDate)
    return calendar

def get_calendar_dates2(effective_dates, range_freq = 'BM'):
    Mindate = effective_dates[0]
    calendar_DTIndex = pd.date_range(start = Mindate, end = pd.to_datetime('today').strftime('%Y-%m-%d'), freq = range_freq)
    calendar = calendar_DTIndex.strftime('%Y-%m-%d')
    calendar = pd.DataFrame(data = {'ExtendedDates' : calendar})
    
    df_effective_dt = pd.DataFrame(data = {'ScdDataDate' : effective_dates})
    
    calendar = calendar.merge(df_effective_dt, left_on = 'ExtendedDates', right_on = 'ScdDataDate', how = 'left')
    calendar['ScdDataDate'] = calendar['ScdDataDate'].fillna(method = 'ffill', axis = 0)
    calendar.dropna(how = 'any', inplace = True)
    
    calendar.ExtendedDates = pd.to_datetime(calendar.ExtendedDates)
    calendar.ScdDataDate = pd.to_datetime(calendar.ScdDataDate)
    return calendar

def create_subuniverse(SectorId='IG000BA008', COC='US', MinDate = '2005-01-01'):
    """ 
        Function to curate the dependent variable 
        Can handle any carve out of the Glbl Markets given SectorId, COC, MinDate options
        
        Args:
        SectorId (str)  : SectorId
        COC (str)    : COC
        MinDate (str) : Date from which you want the universe to be created
        Returns:
        SQLTableName : Table having subuniverse carveout with month end dates
    
    """ 
    
    #pdb.set_trace()
    #query = """ select a.PortfolioEffectiveDate, DataDate, PerformanceId, CompanyId, a.ThirdPartyId, SectorId, SizeSegment, a.TotalOutstandingShares, a.FloatFactor, b.Date, b.LocalClose from Scd..xv_GlobalMarketsRebalanceUniverse a join TimeSeries..EquityDailyData b on a.PerformanceId = b.ShareClassId where b.Date in """ + str(calendar) +" and SectorId = '" + SectorId + """' and COC = '""" + COC +"""' and PortfolioEffectiveDate > '""" + MinDate + "' order by a.DataDate, a.PerformanceId, b.Date"
    queryScd =  """ select a.PortfolioEffectiveDate, DataDate, PerformanceId, CompanyId, a.ThirdPartyId, SectorId, SizeSegment, a.TotalOutstandingShares, a.FloatFactor from Scd..xv_GlobalMarketsRebalanceUniverse a where SectorId in """ + SectorId + """ and COC = '""" + COC +"""' and PortfolioEffectiveDate > '""" + MinDate + "' "
    input_dependent = dc.sqlread(queryScd, 'IDNPDDB61')
    input_dependent.DataDate = pd.to_datetime(input_dependent.DataDate)
    input_dependent = input_dependent.sort_values(by = ['PortfolioEffectiveDate', 'PerformanceId'])
    #calendar = get_calendar_dates(MinDate, range_freq = 'B')
    calendar = get_calendar_dates2(np.unique(input_dependent.PortfolioEffectiveDate.sort_values().dt.strftime("%Y-%m-%d")).tolist(), range_freq = 'B')
    
    input_dependent = input_dependent.merge(calendar, left_on = 'PortfolioEffectiveDate', right_on = 'ScdDataDate', how = 'left')
    if len(dc.sqlread(""" select top 1* from StagingData..SC_temptable """, 'idnpddb61')) > 0:
        print(f"Table exists. Truncating table.")
        db.DatabaseDoQuery("""TRUNCATE TABLE StagingData..SC_temptable""")
        
        print(f"Adding records to table")
        dc.sqlwrite('StagingData..SC_temptable', input_dependent, 'idnpddb61')
    else:           
        
        print(f"Adding records to table")
        dc.sqlwrite('StagingData..SC_temptable', input_dependent, 'idnpddb61')
        
    temp_table_name = 'StagingData..SC_temptable' #name of temptable
    return temp_table_name


def append_subuniverse(table_name, version, SectorId, COC):
    """ 
        Function to curate the dependent variable 
        Can handle any carve out of the Glbl Markets given SectorId, COC, MinDate options
        
        Args:
        SectorId (str)  : SectorId
        COC (str)    : COC
        table_name (str)  : Table to append records to
        version (str)    : version number
        
        Returns:
        SQLTableName : Table having subuniverse carveout till latest day
    
    """ 
    
    #pdb.set_trace()
    #query = """ select a.PortfolioEffectiveDate, DataDate, PerformanceId, CompanyId, a.ThirdPartyId, SectorId, SizeSegment, a.TotalOutstandingShares, a.FloatFactor, b.Date, b.LocalClose from Scd..xv_GlobalMarketsRebalanceUniverse a join TimeSeries..EquityDailyData b on a.PerformanceId = b.ShareClassId where b.Date in """ + str(calendar) +" and SectorId = '" + SectorId + """' and COC = '""" + COC +"""' and PortfolioEffectiveDate > '""" + MinDate + "' order by a.DataDate, a.PerformanceId, b.Date"
    
    max_dates = dc.sqlread("""select max(ExtendedDates), max(PortfolioEffectiveDate) from """ + str(table_name),'idnpddb61')
    max_extendeddt, max_effectivedt = max_dates.iloc[0,0].strftime("%Y-%m-%d"), max_dates.iloc[0,1]
    
    queryScd =  """ select a.PortfolioEffectiveDate, DataDate, PerformanceId, CompanyId, a.ThirdPartyId, SectorId, SizeSegment, a.TotalOutstandingShares, a.FloatFactor from Scd..xv_GlobalMarketsRebalanceUniverse a where SectorId in """ + SectorId + """ and COC = '""" + COC +"""' and PortfolioEffectiveDate = '""" + max_effectivedt+ "' "
    input_dependent = dc.sqlread(queryScd, 'IDNPDDB61')
    input_dependent.DataDate = pd.to_datetime(input_dependent.DataDate)
    input_dependent = input_dependent.sort_values(by = ['PortfolioEffectiveDate', 'PerformanceId'])
    #calendar = get_calendar_dates(MinDate, range_freq = 'B')
    calendar = get_calendar_dates2(np.unique(input_dependent.PortfolioEffectiveDate.sort_values().dt.strftime("%Y-%m-%d")).tolist(), range_freq = 'B')
    
    input_dependent = input_dependent.merge(calendar, left_on = 'PortfolioEffectiveDate', right_on = 'ScdDataDate', how = 'left')
    input_dependent['Version'] = version
    
    input_dependent = input_dependent.loc[input_dependent['ExtendedDates'] > max_extendeddt]
    
    if len(input_dependent) == 0:
        print("Table updated to latest day. Pulling data ...")
        #data = dc.sqlread(""" select * from """ + str(table_name), 'idnpddb61')
        
        return table_name
        
    else:           
        
        print(f"Inserting new dates to table")
        im.DatabaseDoQuery("""insert into """ + str(table_name) + """ values(?,?,?,?,?,?,?,?,?,?,?,?)""", input_dependent, conn=conn)
        
        print("Table updated to latest day. Pulling data ...")
        #data = dc.sqlread(""" select * from """ + str(table_name), 'idnpddb61')
        
        return table_name
        
        
def turnover(ports, groups= None, save_plot = False, dir_name = None):
    """
    Returns period wise turnover numbers and plot
    
    Args:
        *ports(dataframe) : dataframe of portfolios with weights 
                            - if groups are active add weights for every group
                            
                            expected columns format
                            .........................
                            | Date | SecId | Weight | Groups(optional) |
        
        *groups(str) : group the portfolios and run turnover calc
        *save_plot(bool) : whether to save the turnover plot
        *dir_name(str) : path where to save the plot if save_plot set to True
    
    """
    
    if not groups:
        ports.columns = ['Date', 'SecId', 'wts']  
    else:
        ports.columns = ['Date', 'SecId', 'wts', groups]  
    
    ports['Date'] = pd.to_datetime(ports['Date']).dt.strftime('%Y-%m-%d')
    
    act_dates = ports['Date'].unique().tolist()
    act_dates = sorted(act_dates)
    prev_dates = act_dates.copy()
    prev_dates.pop()
    act_dates = sorted(act_dates, reverse = True)
    act_dates.pop()
    act_dates = sorted(act_dates)
    
    prev_ports = ports.loc[ports['Date'] != ports['Date'].max()]
    dict_dates = dict(zip(prev_dates, act_dates))
    
    prev_ports['Date'].replace(dict_dates, inplace = True)
    
    if not groups:
        
        merge_df = ports.loc[ports['Date'] != ports['Date'].min()].merge(prev_ports,\
                   on = ['Date', 'SecId'], how = 'outer', suffixes = ('_curr', '_prev'))

        merge_df.fillna(0, inplace = True)
        merge_df['sec_turnover'] = np.where(merge_df['wts_curr'] - merge_df['wts_prev'] > 0,\
                                            merge_df['wts_curr'] - merge_df['wts_prev'], 0)

        #merge_df.to_csv(r'merge_df_turnover2.csv', index = False)

        turnover = merge_df.groupby('Date')['sec_turnover'].sum().reset_index(drop = False).\
                   rename(columns = {'sec_turnover' : 'Turnover'})
        
        display(turnover)
    
        #year_turnover = turnover.groupby(pd.to_datetime(ports['Date']).dt.year)['Turnover'].sum().reset_index(drop = False).\
        #           rename(columns = {'sec_turnover' : 'Annual_Turnover'})
        #display(year_turnover)


        pic_turnover = turnover.set_index('Date').plot(figsize = (20,15), grid = True, title = 'Turnover number plot')

        turn_plot = pic_turnover.get_figure()
    
    else:
        
        merge_df = ports.loc[ports['Date'] != ports['Date'].min()].merge(prev_ports,\
                   on = ['Date', 'SecId', groups], how = 'outer', suffixes = ('_curr', '_prev'))
        
        #merge_df.to_csv(r'merge_df_turnover3.csv', index = False)
        
        merge_df['wts_curr'].fillna(0, inplace = True)
        merge_df['wts_prev'].fillna(0, inplace = True)
        
        merge_df['sec_turnover'] = np.where(merge_df['wts_curr'] - merge_df['wts_prev'] > 0,\
                                            merge_df['wts_curr'] - merge_df['wts_prev'], 0)

        turnover = merge_df.groupby(['Date', groups])['sec_turnover'].sum().reset_index(drop = False).\
                   rename(columns = {'sec_turnover' : 'Turnover'})
        
        turnover2 = pd.pivot_table(turnover, index = 'Date', columns = groups, values = 'Turnover', aggfunc = 'mean')
        
        display(turnover2)
        
        turnover['Date'] = pd.to_datetime(turnover['Date'])
        
        #cols = 2
        #rows = np.ceil(len(ports[groups].unique())/cols).astype(int)
        rows = np.ceil(len(ports[groups].unique())).astype(int)

        fig, axs = plt.subplots(np.ceil(len(ports[groups].unique())).astype(int) ,figsize=(15,20))
        fig.suptitle('Turnover Plots for different groups')
        # Make space for and rotate the x-axis tick labels
        #fig.autofmt_xdate()

        fig2, ax2 = plt.subplots(figsize=(15,15))
        fig2.suptitle('Single Turnover Plot for different groups')
        
        
        
        counter = 0
        group_list = ports[groups].sort_values().unique().tolist()


        for iter1 in np.arange(0, rows):
            
            if iter1+1 > len(group_list):
                break
            else:
                group = group_list[iter1]
                #print(group)

                turn_ax = turnover.loc[turnover[groups] == group]
                turn_ax.sort_values('Date')
                axs[iter1].plot(turn_ax['Date'], turn_ax['Turnover'])
                axs[iter1].set_title('Group: ' + str(group))
                
                ax2.plot(turn_ax['Date'], turn_ax['Turnover'], label = str(group))
                ax2.legend()
                
        plt.show()
        
        for ax in axs.flat:
            ax.set(xlabel='Date', ylabel='Turnover')
    

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    #for ax in axs.flat:
    #    ax.label_outer()
    
    #plt.show()
            
    '''save plot doesn't work for groups yet'''

    if save_plot:
        save(turn_plot, dir_name, str(pic_turnover.title), stype = 'png')
    
    if not groups:
        return turnover
    else:
        return turnover2

def choose_first_k_stocks(ser,k):
    ser.iloc[:k] = 1
    ser.iloc[k:] = 0
    return ser
 
def cleantable(df, subset):
    df = df.dropna(subset = ['Date'], how = 'any')
    dates = sorted(df.Date.unique().tolist())
    df1 = df[df.Date > dates[-7]]
    subset2 = subset
    subset2.pop(-1)
    df1 = df1.dropna(subset = subset2, how = 'any')
    df2 = df[df.Date <= dates[-7]]
    df2 = df2.dropna(subset = subset, how = 'any')
    df = pd.concat([df2,df1])
    df.sort_values(inplace = True, by = 'Date', ascending = False)
    return df