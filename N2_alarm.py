# 2022.04.12 N2_alarm.py
import datetime
import urllib
import pandas as pd

__author__ =  'Keita Tanaka'
__version__=  '1.0.0' #2022.04.12

print('===============================================================================')
print(f"Noise Analysis of Transition Edge Sensor ver {__version__}")
print(f'by {__author__}')
print('===============================================================================')

class AlarmSetting:
  def __init__(self):
    pass

  def date(self):
    time = datetime.datetime.now()
    print(time)

  def load_csv(self):
    link = "https://yamasakilab.sharepoint.com/:x:/s/K.Tanaka/EU5QlLv9e45Ov4h1VOIMg4EBRr-czgAiHecdGvD9PUrm4A?e=qTOkbD"
    headers = { "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0" }  
    req = urllib.request.Request(link,headers=headers)
    file = urllib.request.urlopen(req)
    #xls  = pd.ExcelFile(file)
    print(file)