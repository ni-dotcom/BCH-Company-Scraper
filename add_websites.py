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
TO_AVOID = [r"/blog/", r"^blog\.", r"/blogs/", r"\.blog/", r"wiki", r"pedia."]  # urls that contain these should be avoided

# Filter URLs through AI
def classify_results(results, name):
  client = Groq(api_key=GROQ_API_KEY)

  items = [{"query": name, "i": i, "title": r["title"], "snippet": r.get("body", "")}
            for i, r in enumerate(results)]

  prompt = f"""For each search result, classify if it is the website of the company/organization that was queried. If none of
  the website titles represent the official site of the organization, choose a tentative one that seems to have the most information about
  the organization. Respond ONLY with a JSON array of integers, where 0 represents the official website and 1 represents a 
  tentative website, in the same order as the input. Finally, add an additional integer at the end that marks whether
  or not the search query even represents a company/organization, where 0 means it does and 1 means a non-organization (i.e. a person or award).
Results: {json.dumps(items)}"""

  try:
    chat_completion = client.chat.completions.create(
      messages=[{"role": "user", "content": prompt,}],
      model="llama-3.3-70b-versatile",
    )

    classified = json.loads(chat_completion.choices[0].message.content)
    print(classified)

    # pair together each result with its bool, and return those with False (are relevant)
    official = [r for r, response in zip(results, classified) if response == 0]
    tentative = [r for r, response in zip(results, classified) if response == 1]
    skip = True if classified[-1]==1 else False

    return official, tentative, skip

  except Exception: # also if API key is exceeds free limit
    return [], results, False

# Use DDG to find links
def find_urls(name):
  try:
    # Run a text search and limit to 5 results
    results = ddgs.text(f"{name} company organization", max_results=5)

    # filters out results that have certain terms
    filtered = list(filter(lambda result: not any(re.search(p, result["href"]) for p in TO_AVOID), results))

    filtered, tentative, skip = classify_results(filtered, name)

    if not skip:
      result = filtered[0] if filtered else None

      print(f"{name} gives {result}")
      url = result.get("href") if result else None
      more_url = tentative[0].get("href") if tentative else None

      if url:
        return url, None, None
      else:
        return url, more_url, None

    else:
      print(f"{name} should be skipped")
      return None, None, "Yes"

  except Exception as e:
    print(e)
    return None, None, None

df[["Website", "Tentative Website", "Non-organization?"]] = df["Name"].apply(find_urls).apply(pd.Series)
not_found = df[df["Website"].isna() & df["Tentative Website"].isna() & df["Non-organization?"].isna()][["Name", "Primary Key", "Website", "Tentative Website"]].copy()
skip = df[df["Non-organization?"].notna()][["Name", "Primary Key", "Non-organization?"]].copy()

# Write to Excel
output = "Websites_given.xlsx"
with pd.ExcelWriter(output, engine='openpyxl', mode='w') as writer:
  df[["Name", "Primary Key", "Website", "Tentative Website", "Non-organization?"]].to_excel(writer, sheet_name="Summary", index=False)
  not_found.to_excel(writer, sheet_name="Not found", index=False)
  skip.to_excel(writer, sheet_name="To Skip", index=False)
  df.to_excel(writer, sheet_name="All Data", index=False)

files.download(output)
