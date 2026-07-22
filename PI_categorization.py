from google.colab import files
import pandas as pd
import numpy as np
import curl_cffi.requests
import requests
import re
from ddgs import DDGS
from bs4 import BeautifulSoup
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# pull from excel
df = pd.read_excel("Contact_BCH_1.xlsx")

def pull_pi(title):
  good_titles = ["professor", "researcher", "scientist", "director", "investigator", "associate", "chief", "instructor", "faculty", "attending"]
  if any(word in title.lower() for word in good_titles):
    return title
  else:
    return None

df["Title"] = df["Title"].apply(pull_pi)
not_pi = df[df["Title"].isna()]
df = df[df["Title"].notna()]

# config
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

site = "site:research.childrenshospital.org"
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
    "Antibody": ["antibody", "polyclonal", "lot-to-lot variability", "cross-reactivity"],
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

def combine_name(first, middle, last):
  if middle is not np.nan:
    return first + f" {middle}" + f" {last}"
  return first + f" {last}"

def find_website(name):
  try:
    result = ddgs.text(f"{name} {site}", max_results=1)
    result = result[0].get("href")

    if "researchers" not in result:
      raise Exception("PI not found")
    
    return result
    
  except Exception as e:
    print(e)
    return None

def fix_url(url):
  if url.startswith('http://'):
    return url.replace("http://", "https://")
  return url

def scrape_research_overview(url):
    try:
      headers = {"User-Agent": "Mozilla/5.0"}
      response = session.get(url, headers=headers, timeout=8, allow_redirects=True)
      response.raise_for_status()

      soup = BeautifulSoup(response.text, "html.parser")

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

# scrape the "Research Overview" section of the first link that shows up
def scrape_publications(urls):
  try:
    if not urls:
      raise Exception("no publication urls found")
    for url in urls:
      print(url)
      response = curl_cffi.requests.get(url, impersonate="chrome")
      response.raise_for_status()

      soup = BeautifulSoup(response.text, "html.parser")

      scraped_text = []

      article = soup.find(id='article-details')
      if article:
        if article.find(id='heading'):
          article.find(id='heading').decompose()
        text = article.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).lower().strip()
        scraped_text.append(text)

      scraped_text = list(dict.fromkeys(scraped_text))  # De-duplicate repeated sections
      combined = "\n".join(scraped_text)

      matched = set()
      keys = set()

      # general keyword matching
      for kw, cat in flat_keyword_map.items():
          if kw in combined:
              matched.add(cat)
              keys.add(kw)
      
      matched_str = ", ".join(sorted(matched)) if matched else np.nan
      keys_str = ", ".join(sorted(keys)) if keys else np.nan
      return matched_str, keys_str, combined

  except Exception as e:
    print(e)
    return None, None, None

df["Name"] = df[["First Name", "Middle Initial", "Last Name"]].apply(lambda names: combine_name(*names), axis=1)
df["Main Website"] = df["Name"].apply(find_website)
df[["Web_cat", "web_kw", "publication_links"]] = df["Main Website"].apply(scrape_research_overview).apply(pd.Series)
df[["Pub_cat", "pub_kw", "pub_text"]] = df["publication_links"].apply(scrape_publications).apply(pd.Series)

output = "PIs_found.xlsx"
with pd.ExcelWriter(output, engine='openpyxl', mode='w') as writer:
  df.to_excel(writer, sheet_name="PIs_categorized", index=False)
  not_pi.to_excel(writer, sheet_name="Not a PI", index=False)

files.download(output)
