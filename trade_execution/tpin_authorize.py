# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 06:22:30 2022

@author: schoudh
"""

credentials_json = r'C:/Users/schoudh/Documents/Modeling/Krauss RAF Momentum Model/gmail_inbox_data_credentials.json'

# import the required libraries
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import os.path
import base64
import os
from kiteconnect import KiteConnect
from kiteconnect import KiteTicker
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time, pyotp
#import undetected_chromedriver as uc
import re
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from chromedriver_py import binary_path

os.chdir(r'C:\Users\schoudh\Documents\Modeling\Krauss RAF Momentum Model')

# Define the SCOPES. If modifying it, delete the token.pickle file.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_tpin():
    
    #Variable creds will store the user access token.
       	# If no valid token found, we will create one.
    creds = None
       
       	# The file token.pickle contains the user access token.
       	# Check if it exists
    if os.path.exists('token.pickle'):
       
       		# Read the token from the file and store it in the variable creds
       		with open('token.pickle', 'rb') as token:
       			creds = pickle.load(token)
       
       	# If credentials are not available or are invalid, ask the user to log in.
    if not creds or not creds.valid:
       		if creds and creds.expired and creds.refresh_token:
       			creds.refresh(Request())
       		else:
       			flow = InstalledAppFlow.from_client_secrets_file(credentials_json, SCOPES)
       			creds = flow.run_local_server(port=0)
       
       		# Save the access token in token.pickle file for the next run
       		with open('token.pickle', 'wb') as token:
       			pickle.dump(creds, token)
       
       	# Connect to the Gmail API
    service = build('gmail', 'v1', credentials=creds)
       
       	# request a list of all the messages
    result = service.users().messages().list(userId='me', maxResults = 50).execute()
       
       	# We can also pass maxResults to get any number of emails. Like this:
       	# result = service.users().messages().list(maxResults=200, userId='me').execute()
    messages = result.get('messages')
    #msg = messages[0]
    #txt = service.users().messages().get(userId='me', id=msg['id']).execute()
    counter = 0
    for msg in messages:
        txt = service.users().messages().get(userId='me', id=msg['id']).execute()
        
        print(f"Counter = {counter} and id = {msg['id']}")
        counter += 1
        
        payload = txt['payload']
        headers = payload['headers']
        
        for d in headers:
            if d['name'] == 'Subject':
                subject = d['value']
                
            if d['name'] == 'From':
                sender = d['value']
                
            if d['name'] == 'Date':
                datetime = d['value']
        
        if counter > 15:
            print('Cannot find match. Need to re-run search for otp.')
            return None
        else:        
            print('Match... Extracting Details')
            
        if subject == 'Transaction OTP':
                pass
        else:
                continue
    		
        if sender == 'edis@cdslindia.co.in':
                pass
        else:
                continue
            
        print(datetime)            
        if np.timedelta64(pd.to_datetime(pd.to_datetime('now').tz_localize('UTC').tz_convert('Asia/Kolkata').strftime('%Y-%m-%d %H:%M:%S'))\
                          - pd.to_datetime(datetime[:-12]), 's').astype(int) < 120:
            pass
        else:
            continue
    
        
        break
    
    msg2 = txt['snippet']
    newmsg = ''.join(msg2.split())
    otp = (re.findall(r'is:(\d+)', newmsg)[0])
    
    return otp

def tpin_verify(api_key, api_secret, user_id, user_pwd, totp_key):
    driver = webdriver.Chrome(executable_path=binary_path)
    driver.get(r'https://kite.zerodha.com/holdings')
    login_id = WebDriverWait(driver, 10).until(lambda x: x.find_element_by_xpath('//*[@id="userid"]'))
    login_id.send_keys(user_id)

    pwd = WebDriverWait(driver, 10).until(lambda x: x.find_element_by_xpath('//*[@id="password"]'))
    pwd.send_keys(user_pwd)
    
    submit = driver.find_element(By.CLASS_NAME, 'actions')
    #submit = WebDriverWait(driver, 10).until(lambda x: x.find_element_by_xpath('//*[@id="container"]/div/div/div[2]/form/div[4]/button'))
    submit.click()

    time.sleep(1)
    #adjustment to code to include totp
    totp = WebDriverWait(driver, 10).until(lambda x: x.find_element_by_xpath('//*[@label="External TOTP"]'))
    authkey = pyotp.TOTP(totp_key)
    totp.send_keys(authkey.now())
    #adjustment complete
    
    continue_btn = driver.find_element(By.CLASS_NAME, 'actions')
    #continue_btn = WebDriverWait(driver, 10).until(lambda x: x.find_element_by_xpath('//*[@id="container"]/div/div/div[2]/form/div[3]/button'))
    continue_btn.click()

    time.sleep(1)
    
    auth_tag = driver.find_element(By.LINK_TEXT, 'Authorisation')
    auth_tag.click()
    
    time.sleep(1)
    
    auth_confirm = driver.find_element(By.XPATH, "//button[@type = 'button'][@class = 'button button-blue']")
    auth_confirm.click()
    
    time.sleep(1)
    
    auth_confirm2 = driver.find_element(By.XPATH, "//button[@type = 'button'][@class = 'button button-blue']")
    auth_confirm2.click()
    
    time.sleep(1)
    
    # Switching to CDSL page
    cdsl_window = driver.window_handles[1]
    driver.switch_to.window(cdsl_window)
    driver.implicitly_wait(120)
    time.sleep(3)
    
    try:
        WebDriverWait(driver, 8).until(EC.presence_of_element_located((By.XPATH, "/html/body/div[1]/div/div/div[2]/div[2]/button")))
        # Selecting "Continue to CDSL"
        driver.find_element_by_xpath("/html/body/div[1]/div/div/div[2]/div[2]/button").click()
    except TimeoutException:
        print("CDSL Page Not Loaded")

    # Entering TPIN
    driver.find_element_by_id("txtPIN").send_keys('017111')
    driver.find_element_by_id("btnCommit").click()
    driver.implicitly_wait(60)
    
    time.sleep(40)
    
    print('Getting otp now.')
    otp = get_tpin()
    
    if not otp:
        otp = get_tpin()
    else:
        print('We have the tpin otp: ', otp)
    
    #driver.switch_to.window(cdsl_window)
    #driver.implicitly_wait(60)
    
    driver.find_element_by_id("OTP").send_keys(otp)
    driver.implicitly_wait(60)
    driver.find_element_by_id("VerifyOTP").click()
    driver.implicitly_wait(60)
    time.sleep(20)
    print("Success")
    driver.quit()
    
    return True

verify = tpin_verify(api_key = "qojtsll8rpdcbgza", api_secret = "xz6nf3lhru1rpqwa9ej44r4eb19qaw2w", user_id = 'DB3408', user_pwd = '@lPhaBet123', totp_key = 'K3A5K3664XZH4L6VNJS4VJBX4ZHZMUST')

# =============================================================================
# otp = get_tpin()
# otp
# =============================================================================



