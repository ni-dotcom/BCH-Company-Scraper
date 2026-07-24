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
from playwright.async_api import async_playwright

# config
NIH_KEY = userdata.get('NIH_API')
BASE = "https://research.childrenshospital.org"
START = f"{BASE}/researchers"

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

async def collect_links():
    researcher_urls = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto(START, wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)

        last_height = 0

        while True:
            anchors = await page.locator('a[href*="/researchers/"]').evaluate_all(
                "els => els.map(a => a.href)"
            )

            for href in anchors:
                if href.startswith(f"{BASE}/researchers/") and href.rstrip("/") != START.rstrip("/"):
                    researcher_urls.add(href.rstrip("/"))

            print(f"Found so far: {len(researcher_urls)}")

            new_height = await page.evaluate("document.body.scrollHeight")
            if new_height == last_height:
                break

            last_height = new_height
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2500)

        await browser.close()

    return sorted(researcher_urls)

# scrape the "Research Overview" section of the first link that shows up
def scrape_research_overview(url):
    try:
      headers = {"User-Agent": "Mozilla/5.0"}
      response = session.get(url, headers=headers, timeout=8, allow_redirects=True)
      response.raise_for_status()

      soup = BeautifulSoup(response.text, "html.parser")

      # Extract the PI's name
      title = soup.find('title').get_text()
      name = title.split('|')[0].strip()
      name = re.sub(r",?\s*(MD|PhD|MPH|MS|RN|DO|MSc|ScD)\b\.?", "", name).strip()
      first, last = name.rsplit(" ", maxsplit=1)
      middle = ""
      if "." in first:
        first, middle = first.rsplit(" ", maxsplit=1)
        middle = middle.removesuffix(".")

      # prevent ending code if overview doesn't exist
      overview_div = soup.find("div", id="overview")
      research_overview = overview_div.get_text(" ", strip=True).lower() if overview_div else ""
      print(name, research_overview)

      matched = set()
      keys = set()

      # general keyword matching
      for kw, cat in flat_keyword_map.items():
          if kw in research_overview:
              matched.add(cat)
              keys.add(kw)   

      urls = []
      publications = soup.find("div", id="publications")
      if publications:
          links = publications.find_all("li")
          for link in links:
              if any(year in link.get_text(" ", strip=True) for year in ["2026", "2025", "2024", "2023"]):
                  a = link.find("a", href=True)
                  if a:
                      urls.append(a["href"])
      
      matched_str = ", ".join(sorted(matched)) if matched else np.nan
      keys_str = ", ".join(sorted(keys)) if keys else np.nan
      # urls_str = ", ".join(sorted(urls)) if urls else np.nan
      return first.strip(), middle.strip(), last.strip(), matched_str, keys_str, urls

    except Exception as e:
      print(f"scrape_research_overview error: {e}")
      return None, None, None, None, None, None

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
    print(f"scrape_publications error: {e}")
    return None, None, None

# df["BCH Webpage"] = df[["First Name", "Middle Initial", "Last Name"]].apply(lambda names: find_website(*names), axis=1)
# not_found = df[df["BCH Webpage"].isna()]
# df = df[df["BCH Webpage"].notna()]

urls = await collect_links()
print(f"Found {len(urls)} URLs")
df = pd.DataFrame({"BCH Webpage": urls})
df[["First Name", "Middle Initial", "Last Name", "Categories (from BHC webpage)", "Keywords from BCH webpage", "Links to Publications"]] = df["BCH Webpage"].apply(scrape_research_overview).apply(pd.Series)

pi_list = pd.read_excel("Contacts_BCH.xlsx")

for frame in [pi_list, df]:
    frame["First Name"] = frame["First Name"].astype(str).str.strip()
    frame["Last Name"] = frame["Last Name"].astype(str).str.strip()
new_pi_list = pd.merge(pi_list, df, on=["First Name", "Last Name"])
new_pi_list[["Categories (from publications)", "Publication Keywords", "Scraped Text from Publications"]] = new_pi_list["Links to Publications"].apply(scrape_publications).apply(pd.Series)

output = "PIs_found.xlsx"
with pd.ExcelWriter(output, engine='openpyxl', mode='w') as writer:
  df.to_excel(writer, sheet_name="PIs Categorized (Summary)", index=False)
  pi_list.to_excel(writer, sheet_name="Old PI List", index=False)
  new_pi_list.to_excel(writer, sheet_name="New PI list", index=False)
  new_pi_list[["First Name", "Last Name", "Department", "Categories (from BHC webpage)", "Categories (from publications)"]].to_excel(
      writer, sheet_name="(Summary)", index=False)
  # not_pi.to_excel(writer, sheet_name="Not a PI", index=False)

files.download(output)
