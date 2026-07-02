from ddgs import DDGS
from google.colab import files
import pandas as pd
import requests

ddgs = DDGS(api_url="http://localhost:4479", spawn_api=True)

df = pd.read_excel("Companies_NeedsWebsites_test.xlsx")

def find_urls(name):
  if isinstance(name, str):
    try:
      search = ddgs.text(f"{name} -site:wikipedia.org", max_results=1)
      print(f"{name} gives {search[0]}")
      url = search[0].get("href")
      return url
    except Exception as e:
      print(e)
      return None
  else:
    print(f"{name} isn't a string")
    return None

df["Website"] = df["Name"].apply(find_urls)
not_found = df[df["Website"].isna()].copy()

output = "Websites_given.xlsx"
with pd.ExcelWriter(output, engine='openpyxl', mode='w') as writer:
  df.to_excel(writer, sheet_name="All", index=False)
  not_found.to_excel(writer, sheet_name="Not found", index=False)
