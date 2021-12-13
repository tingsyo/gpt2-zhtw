#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
This script retrieves processed PTTGOSSIP data with API.
'''

from __future__ import print_function
import pandas as pd
import requests
from datetime import datetime, timedelta

# Parameters
BASE_URL = 'http://52.26.156.12:4001/get_json/?'
TODAY = datetime.today()
DATA_DATE= datetime.strftime(TODAY-timedelta(days=1), '%Y-%m-%d')

def retrieve_pttgossip(base_url=BASE_URL, data_date=None):
    if not data_date is None:
        date_string = '&item_date='+data_date
    else:
        date_string = ''
    #
    posts = requests.get(base_url+'item_type=posts'+date_string)
    tfreq = requests.get(base_url+'item_type=termscores'+date_string)
    # Convert to DataFrame
    posts = pd.DataFrame(eval(posts.text))
    tfreq = pd.DataFrame(eval(tfreq.text))
    return((posts, tfreq))

# Main script
posts, tfreq = retrieve_pttgossip(base_url=BASE_URL, data_date=DATA_DATE)
posts.to_csv('posts.csv', index=False)
tfreq.to_csv('tfreq.csv', index=False)
