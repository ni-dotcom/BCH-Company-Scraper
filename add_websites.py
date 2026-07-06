from ddgs import DDGS
from google.colab import files, userdata
import pandas as pd
import requests
import re
from groq import Groq
import json

# CONFIG
df = pd.read_excel("Companies_NeedsWebsites_test.xlsx")
ddgs = DDGS(api_url="http://localhost:4479", spawn_api=True)
GROQ_API_KEY = userdata.get('GROQ_API_KEY')
TO_AVOID = [r"/blog/", r"^blog\.", r"/blogs/", r"\.blog/", r"wiki", r"pedia."]

def skip_urls(url): # make this lambda
  return any(re.search(p, url) for p in TO_AVOID)

# Filter URLs through AI
def classify_results(results, name):
  client = Groq(api_key=GROQ_API_KEY)

  items = [{"query": name, "i": i, "title": r["title"], "snippet": r.get("body", "")} 
            for i, r in enumerate(results)]

  prompt = f"""For each search result, classify if it's a personal/informal blog and/or simply unrelated to the query 
(vs. a company or organizational website about the query). Respond with ONLY a JSON array of 
booleans, same order as input, true = is a blog or is unrelated.
Results: {json.dumps(items)}"""

  chat_completion = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt,}],
    model="llama-3.3-70b-versatile",
  )

  is_irrelevant = json.loads(chat_completion.choices[0].message.content)

  # pair together each result with its bool, and return those with False (are relevant)
  return [r for r, irrelevant in zip(results, is_irrelevant) if not irrelevant]

# Use DDG to find links
def find_urls(name):
  if isinstance(name, str):
    try:
      results = ddgs.text(f"{name} company organization", max_results=5)  # search DDG for company name and get 5 links
      filtered = [r for r in results if not skip_urls(r["href"])]

      filtered = classify_results(filtered, name) # comment out this line to not use AI


      result = filtered[0] if filtered else None  # get first link

      print(f"{name} gives {result}")
      url = result.get("href")
      return url

    except Exception as e:
      print(e)
      return None
  
  else:
    print(f"{name} isn't a string")
    return None

df["Website"] = df["Name"].apply(find_urls)
not_found = df[df["Website"].isna()].copy()

# Write to Excel
output = "Websites_given.xlsx"
with pd.ExcelWriter(output, engine='openpyxl', mode='w') as writer:
  df.to_excel(writer, sheet_name="All", index=False)
  not_found.to_excel(writer, sheet_name="Not found", index=False)

files.download(output)
