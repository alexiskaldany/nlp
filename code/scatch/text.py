# url = "https://www.sec.gov/Archives/edgar/data/103872/000010387222000007/a10312021ex-21.htm"
# headers={
#             "User-Agent": "exrysadsa@codewalla.com",
#             "Accept-Encoding": "gzip, deflate",
#             "Host": "www.sec.gov",
#         }
import requests
import re
import bs4 as bs
import pandas as pd 
# req = requests.get(url,headers=headers).text
# text = bs.BeautifulSoup(req,'lxml').text
# print(text)

# df = pd.read_html(req)[0]
# df.to_csv('test.csv')
# print(len(df))

path = "/Users/alexiskaldany/school/nlp/data (3).csv"

df = pd.read_csv(path)

contract_ids = df['contract_id'].to_list()
print(len(list(set(contract_ids))))

amaechi_path = '/Users/alexiskaldany/Downloads/Contracts-ProjectNames_party_names_clean.csv'

df = pd.read_csv(amaechi_path)
ids = df['id'].unique()
print(len(ids))