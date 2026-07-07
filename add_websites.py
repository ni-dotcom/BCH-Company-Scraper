from ddgs import DDGS
from google.colab import files, userdata
import pandas as pd
import requests
import re
from groq import Groq
import json

# CONFIG
df = pd.read_excel("Companies_NeedsWebsites_test.xlsx")

GROQ_API_KEY = userdata.get('GROQ_API_KEY')
TO_AVOID = [r"/blog/", r"^blog\.", r"/blogs/", r"\.blog/", r"wiki", r"pedia."]  # urls that contain these should be avoided

# Filter URLs through AI
def classify_results(results, name):
  client = Groq(api_key=GROQ_API_KEY)

  items = [{"query": name, "i": i, "title": r["title"], "snippet": r.get("body", "")} 
            for i, r in enumerate(results)]

  prompt = f"""For each search result, classify if it's a personal/informal blog and/or simply unrelated to the query 
(vs. a company or organizational website about the query). Respond with ONLY a JSON array of 
booleans, same order as input, true = is a blog or is unrelated.
Results: {json.dumps(items)}"""

  try:
    chat_completion = client.chat.completions.create(
      messages=[{"role": "user", "content": prompt,}],
      model="llama-3.3-70b-versatile",
    )

    is_irrelevant = json.loads(chat_completion.choices[0].message.content)

    # pair together each result with its bool, and return those with False (are relevant)
    return [r for r, irrelevant in zip(results, is_irrelevant) if not irrelevant]

  except Exception: # also if API key is exceeds free limit
    return results

def _search_sync(query, max_results=5):
    with DDGS() as ddgs:
        return ddgs.text(query, max_results=max_results)

# Use DDG to find links
async def find_urls(names):
  names = names.tolist()
  urls = []

  for name in names:
    try:
      # Run a text search and limit to 5 results
      results = await asyncio.to_thread(_search_sync, f"{name} company organization", 5)

      filtered = list(filter(lambda result: not any(re.search(p, result["href"]) for p in TO_AVOID), results))  # filters out results that have certain terms

      filtered = classify_results(filtered, name) # comment out this line to not use AI

      result = filtered[0] if filtered else None

      print(f"{name} gives {result}")
      url = result.get("href") if result else None
      urls.append(url)

    except Exception as e:
      print(e)
      urls.append(None)
    
  return urls

df["Website"] = pd.Series(await find_urls(df["Name"]))
not_found = df[df["Website"].isna()].copy()

# Write to Excel
output = "Websites_given.xlsx"
with pd.ExcelWriter(output, engine='openpyxl', mode='w') as writer:
  df.to_excel(writer, sheet_name="All", index=False)
  not_found.to_excel(writer, sheet_name="Not found", index=False)

# files.download(output)
