from ddgs import DDGS
from google.colab import files
import pandas as pd
import requests

ddgs = DDGS(api_url="http://localhost:4479", spawn_api=True)

df = pd.read_excel("Companies_NeedsWebsites.xlsx")

def find_urls(name):
  if isinstance(name, str):
    search = DDGS().text(name, max_results=1)
    url = search.get("href")
    return url
  else:
    return none

df["Website"] = df["Name"].apply(find_urls)
