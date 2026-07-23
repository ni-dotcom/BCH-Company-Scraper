from google.colab import files, userdata
import pandas as pd
import numpy as np
import curl_cffi.requests
import requests
import re
from ddgs import DDGS
from bs4 import BeautifulSoup
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import lxml
from groq import Groq
import json


# pull from excel
df = pd.read_excel("Contact_BCH_1.xlsx")

def pull_pi(title):
  good_titles = ["professor", "researcher", "scientist", "director", "investigator", "associate", "chief", "instructor", "faculty", "attending"]
  try:
    return title if any(word in title.lower() for word in good_titles) else None
  except Exception:
    return None

df["Title"] = df["Title"].apply(pull_pi)
not_pi = df[df["Title"].isna()]
df = df[df["Title"].notna()]

# config
NIH_KEY = userdata.get('NIH_API')
GROQ_API_KEY = userdata.get('GROQ_API_KEY')

ddgs = DDGS(api_url="http://localhost:4479", spawn_api=True)

# requests session with retries
session = requests.Session()
retries = Retry(
    total=1,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "HEAD", "OPTIONS"]
)
session.mount("http://", HTTPAdapter(max_retries=retries))
session.mount("https://", HTTPAdapter(max_retries=retries))

site = "site:research.childrenshospital.org/researchers"
# RESEARCH_TERMS = [
#     "engineer", "discover", "publish", "our-story", "mission",
#     "technology", "platform", "pipeline", "science", "products",
#     "solution", "research"
# ]
# categories (removed modalities)
category_keyword_map = {
    "Allergy" : ["Allergy", "Allergies", "allergen", " IgE ", "anaphylaxis", "urticaria", "hives", "epinephrine auto-injector", "allergic"],
    "Anesthesia": ["Anesthesia", "Anesthetics", "Anesthesiology", "intubation", "analgesia", "sedation", "ASA classification", "perioperative", " PACU "],
    "Autoimmune": ["autoimmune", "lupus", "rheumatoid", "autoantibodies", "multiple sclerosis"],
    "Cardiology": ["cardiology", "arrhythmia", "echocardiogram", "ECG/EKG", "pacemaker", "cardiomyopathy" ,"electrophysiology"],
    "Cardiovascular": ["Cardiovascular", "atherosclerosis", "myocardial ischemia"],
    "Dermatology": ["dermatology", "eczema", "psoriasis", "acne", "dermatitis", "pruritus", "skin biopsy"],
    "Endocrinology": ["Endocrine", "Endocrinology", "adrenal", "thyroid"],
    "Epilepsy": ["epilepsy"],
    "Fetal/Newborn Medicine": ["Newborn", "Newborns", "Fetus", "Fetal", "neonatology", "prenatal diagnosis", "congenital"],
    "Gastroenterology": ["Gastroenterology", "Gastrointestinal", "celiac", "endoscopy", "Crohn's", "pancreatitis", "colonoscopy"],
    "Hematology": ["Hematology", "anemia", "sickle cell", "hemophilia", "thrombocytopenia"],
    "Immunology": ["Immunology", "immune deficiency"],
    "Infectious Disease": ["infectious", "covid", "sars-cov-2", "antivirals", "sepsis"],
    "Metabolic": ["metabolic", "hypoglycemia", "hyperammonemia"],
    "Nephrology": ["Nephrology", "proteinuria", "nephrotic syndrome"],
    "Neurology": ["neurology", "neuron", "epilepsy", "alzheimer", "parkinsons", "Epilepsy","psychiatry", "seizures", "neuropathy", "neuroimaging", "neuromuscular"],
    "Psychiatry": ["psychiatry", "ADHD", "PTSD", "psychosis", "psychopharmacology", "depression", "anxiety", "mental health"],
    "Neuroscience": ["Neuroscience", "brain circuit", "neurodevelopment", "neuroplasticity", "neurogenetics"],
    "Obesity": ["Obesity", "metabolic syndrome", "insulin resistance", "fatty liver"],
    "Oncology": ["oncology", "cancer", "tumor", "carcinoma", "chemotherapy", "leukemia", "lymphoma"],
    "Opthalmology": ["Opthalmology", "amblyopia", "strabismus", "glaucoma", "retina", "cornea", "fundus exam", "eye trauma"],
    "Orthopedics": ["Orthopedic", "sports injury", "ligament/ACL", "joint pain", "hip dysplasia", "physical therapy"],
    "Pulmonary": ["pulmonary", "asthma", "COPD", "cystic fibrosis", "bronchiolitis"],
    "Radiology": ["radiology", "radiologists"],
    "Rare Diseases": ["rare disease", "orphan", "ultra-rare", "diagnostic odyssey"],
    "Reproductive Health": ["Reproductive Health", "Feminine Health", "PCOS"],
    "Surgery": ["laparoscopic"],
    "Transplant":["Transplant", "organ allocation", "HLA matching"],
    "Urology": ["Urology", " UTI ", "kidney stones", "urodynamics"],

    "Biomarkers": ["Biomarkers", "surrogate endpoint", "ROC/AUC"],
    "Diagnostics": ["diagnostic", "diagnostics", "lab-developed test", "clinical utility"],
    "Educational/Training Materials": ["Educational Materials", "Training Materials", "Training and Education", "instructional design", "curriculum"],
    "Medical Devices": ["Medical Devices", "Medical Technology", "implant", "FDA 510(k)", " PMA "],
    "Medical Equipment": ["Medical Equipment", "electrical safety"],
    "Research Tools" : ["Research Tools", "Helping Researchers", "Tools for Researchers"],
    # "Antibody": ["antibody", "polyclonal", "lot-to-lot variability", "cross-reactivity"],
    "Antigen": ["Antigen", "immunogen", "hapten", "immunogenicity", "pathogen-associated"],
    "Assay": [" PCR ", "immunoassay", "limit of detection"],
    "Protein (Research Tool)": ["purification", "activity assay", "binding kinetics", "post-translational modifications"],
    "Software": ["interoperability"],
    "Imaging Software": ["imaging software", " DICOM", " PACS "],
}
flat_keyword_map = {
    kw.lower(): cat
    for cat, kws in category_keyword_map.items()
    for kw in kws
}

def classify_results(results, name):
  client = Groq(api_key=GROQ_API_KEY)

  items = [{"query": name, "i": i, "title": r["title"], "snippet": r.get("body", "")}
            for i, r in enumerate(results)]

  prompt = f"""For each search result, classify if it is the website of the name that was queried, or if it regards a different person or general page.
  Respond ONLY with a JSON array of integers, where 0 means the website is of the person and 1 represents a wrong website, in the same order as the input. 
Results: {json.dumps(items)}"""
# make it respond only with the 0 result

  try:
    chat_completion = client.chat.completions.create(
      messages=[{"role": "user", "content": prompt,}],
      model="llama-3.3-70b-versatile",
    )

    classified = json.loads(chat_completion.choices[0].message.content)
    print(classified)

    # pair together each result with its bool, and return those with False (are relevant)
    official = [r for r, response in zip(results, classified) if response == 0]
    # bad = [r for r, response in zip(results, classified) if response == 1]

    return official

  except Exception: # also if API key is exceeds free limit
    return []

def find_website(first, middle, last):
  try:
    if isinstance(middle, str):
      name = first + f" {middle}" + f" {last}"
    else:
      name = first + f" {last}"
    
    print(name)
    results = ddgs.text(f"{name} {site}", max_results=10)
    print(results)

    best_result = classify_results(results, name)
    if not best_result:
      raise Exception("PI not found")
    print(best_result)
    best_result = best_result[0].get("href")
    print(best_result)
    
    return best_result
    
  except Exception as e:
    print(e)
    return None

def fix_url(url):
  if url.startswith('http://'):
    return url.replace("http://", "https://")
  return url

# scrape the "Research Overview" section of the first link that shows up
def scrape_research_overview(url):
    try:
      headers = {"User-Agent": "Mozilla/5.0"}
      response = session.get(url, headers=headers, timeout=8, allow_redirects=True)
      response.raise_for_status()

      soup = BeautifulSoup(response.text, "html.parser")

      # prevent ending code if overview doesn't exist
      research_overview = soup.find('div', id='overview').text
      research_overview = re.sub(r'\s+', ' ', research_overview).lower().strip()
      print(research_overview)

      matched = set()
      keys = set()

      # general keyword matching
      for kw, cat in flat_keyword_map.items():
          if kw in research_overview:
              matched.add(cat)
              keys.add(kw)   

      urls = []
      publications = soup.find('div', id='publications')
      links = publications.find_all('li')
      for link in links:
        if any(year in link.text for year in ["2026", "2025", "2024", "2023"]):
          url = link.find('a').get("href")
          urls.append(fix_url(url))
      
      matched_str = ", ".join(sorted(matched)) if matched else np.nan
      keys_str = ", ".join(sorted(keys)) if keys else np.nan
      # urls_str = ", ".join(sorted(urls)) if urls else np.nan
      return matched_str, keys_str, urls

    except Exception as e:
      print(e)
      return None, None, None

# refactor out
def extract_pmid(url):
    match = re.search(r'\d+$', url)
    return match.group() if match else None

def parse_article(article):
    pmid = article.PMID.text if article.PMID else None
    title = article.ArticleTitle.text if article.ArticleTitle else ""
    abstract = " ".join(t.text for t in article.find_all("AbstractText"))
    mesh_terms = [m.text for m in article.find_all("DescriptorName")]
    author_keywords = [k.text for k in article.find_all("Keyword")]

    full_text = "\n".join([title, abstract] + mesh_terms + author_keywords).lower()

    return full_text

def scrape_publications(urls):
  try:
    if not urls:
      raise Exception("no publication urls found")
    
    pmids = [extract_pmid(u) for u in urls]
    pmids = [p for p in pmids if p]  # drop any that failed to match
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "api_key": NIH_KEY,
    }

    r = curl_cffi.requests.post(url, params=params)
    r.raise_for_status()
    xml_text = r.text

    soup = BeautifulSoup(xml_text, "xml")

    records = [parse_article(a) for a in soup.find_all("PubmedArticle")]
    print("pmids:", pmids)
    print("status:", r.status_code)
    print(xml_text[:500])
    print("num records parsed:", len(records))

    matched = set()
    keys = set()

    for rec in records:
      # general keyword matching
      for kw, cat in flat_keyword_map.items():
        if kw in rec:
          matched.add(cat)
          keys.add(kw)

    matched_str = ", ".join(sorted(matched)) if matched else np.nan
    keys_str = ", ".join(sorted(keys)) if keys else np.nan
    records = ", ".join(sorted(records)) if records else np.nan
    return matched_str, keys_str, records

  except Exception as e:
    print(e)
    return None, None, None

df["BCH Webpage"] = df[["First Name", "Middle Initial", "Last Name"]].apply(lambda names: find_website(*names), axis=1)
not_found = df[df["BCH Webpage"].isna()]
df = df[df["BCH Webpage"].notna()]

df[["Categories (from BHC webpage)", "Keywords from BCH webpage", "Links to Publications"]] = df["BCH Webpage"].apply(scrape_research_overview).apply(pd.Series)
df[["Categories (from publications)", "Publication Keywords", "Scraped Text from Publications"]] = df["Links to Publications"].apply(scrape_publications).apply(pd.Series)

output = "PIs_found.xlsx"
with pd.ExcelWriter(output, engine='openpyxl', mode='w') as writer:
  df[["First Name", "Middle Initial", "Last Name", "Department", "Categories (from BHC webpage)", "Categories (from publications)"]].to_excel(
      writer, sheet_name="PIs Categorized (Summary)", index=False)
  not_found.to_excel(writer, sheet_name="PI's BCH Webpage Not Found")
  df.to_excel(writer, sheet_name="PIs Categorized (Full Data)", index=False)
  not_pi.to_excel(writer, sheet_name="Not a PI", index=False)

files.download(output)
