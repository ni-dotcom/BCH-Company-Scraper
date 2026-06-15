import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
import re
from google.colab import files

# Ensure the global 'driver' object is accessible. It should have been initialized in previous cells.
# If driver is not initialized, the scraping will fall back to requests.get (can't scrape dynamic content).
if 'driver' not in globals() or driver is None:
    print("Warning: Selenium WebDriver is not initialized. Falling back to requests.get. Dynamic content may not be scraped.")
    # Optionally, can try to re-initialize driver here, but it's better to ensure prior cells run.

# Load Excel
df = pd.read_excel("Companies.xlsx")
df = df.dropna(subset=["Website"]).copy()   # this drops rows w/ empty values in Website column and makes new copy

# Prepend 'https://' to URLs if missing to ensure valid URL format for Selenium and requests
def add_https_if_missing(url):
    if pd.isna(url) or not isinstance(url, str):
        return url
    if not url.startswith('http://') and not url.startswith('https://'):
        return 'https://' + url
    return url

df["Website"] = df["Website"].apply(add_https_if_missing)

# Clean text function
def clean_text(text):
    if isinstance(text, str):
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text) # replaces non-printable ASCII letters with a space
    return text

for col in df.columns:
    if df[col].dtype == 'object': # object data type indicates it contains strings or mixed types
        df[col] = df[col].apply(clean_text)

# Scraping function
def scrape_relevant_sections(url):
    try:
        page_source = ""
        # Use Selenium driver if available, otherwise fallback to requests
        if 'driver' in globals() and driver is not None:
            try:
                driver.get(url)
                page_source = driver.page_source
            except Exception as selenium_e:
                print(f"Selenium failed for {url}: {selenium_e}. Falling back to requests.get.")
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=5)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                page_source = response.text
        else:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            page_source = response.text

        soup = BeautifulSoup(page_source, 'html.parser')  # create soup object which represents html page as nested data structure

        keywords = ["about", "product", "pipeline", "technology", "mission", "vision", "platform", "solution"]
        relevant_texts = [] # List to store individual text sections
        keyword_match_found = False # Flag to track if any keyword sections were found

        for tag in soup.find_all(["h1", "h2", "h3", "a", "strong"]):  # iterates through these html tags
            tag_text = tag.get_text(strip=True).lower() # extracts visible text content from each tag and cleans
            if any(kw in tag_text for kw in keywords):
                parent = tag.find_parent()
                if parent and parent.name not in ["nav", "footer", "header"]: # excludes text from website's nav menu
                    section_text = parent.get_text(separator=' ', strip=True) # text from different child tags are joined with space, and whitespace cleaned
                    section_text_processed = re.sub(r'\s+', ' ', section_text).lower().strip()
                    if len(section_text_processed) > 50:  # more than 50 char
                        relevant_texts.append(section_text_processed)
                        keyword_match_found = True

        # If no important sections are matched by keywords, use fallback
        if not keyword_match_found:
            long_paragraphs = []
            paragraphs = soup.find_all("p")
            for p in paragraphs:
                if len(long_paragraphs) >= 5: # Stop if we already found 5
                    break
                para_text = p.get_text(strip=True)
                para_text_processed = re.sub(r'\s+', ' ', para_text).lower().strip()
                if len(para_text_processed) > 50:
                    long_paragraphs.append(para_text_processed)
            relevant_texts.extend(long_paragraphs) # Take the first 5 long paragraphs

        # Construct the combined string, joining with newlines
        combined_string = "\n".join(relevant_texts)

        if not keyword_match_found and relevant_texts: # If fallback was used and text was found
            combined_string = "fallback: " + combined_string

        # Ensure consistent formatting even if no text was found (return empty string)
        return combined_string.strip()

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return "" # Return an empty string on error


df["Scraped_Text"] = df["Website"].apply(scrape_relevant_sections)

# General keyword-category map (excluding therapies)
category_keyword_map = {
    "Allergy" : ["Allergy", "Allergies"],
    "Anesthesia": ["Anesthesia", "Anesthetics", "Anesthesiology"],
    "Autoimmune": ["autoimmune", "inflammation", "lupus", "rheumatoid","AIDS/HIV"],
    "Cardiology": ["cardiology", "heart", "cardiac", "arrhythmia"],
    "Cardiovascular": ["Cardiovascular","vascular"],
    "Dermatology": ["dermatology", "skin", "eczema", "psoriasis"],
    "Endocrinology": ["Endocrine", "Endocrinology"],
    "Fetal/Newborn Medicine": ["Newborn", "Newborns", "Fetus", "Fetal"],
    "Gastroenterology": ["Gastroenterology", "Gastrointestinal"],
    "Hematology": ["Hematology"],
    "Immunology": ["Immunology", "Immune System"],
    "Infectious Disease": ["infectious", "viral", "bacterial", "covid", "sars-cov-2"],
    "Metabolic": ["metabolic", "obesity", "diabetes", "insulin", "glucose"],
    "Nephrology": ["Nephrology","Kidney"],
    "Neurology": ["neurology", "brain", "neuron", "epilepsy", "alzheimer", "parkinsons", "Epilepsy","psychiatry","depression", "anxiety", "mental health"],
    "Psychiatry": ["psychiatry", "depression", "anxiety", "mental health"],
    "Neuroscience": ["Neuroscience"],
    "Obesity": ["Obesity", "Weight Loss", "Weight Management"],
    "Oncology": ["oncology", "cancer", "tumor", "carcinoma"],
    "Opthamology": ["Opthamology"],
    "Orthopedics": ["Orthopedic"],
    "Pulmonary": ["pulmonary", "respiratory", "lung"],
    "Rare Diseases": ["rare disease", "orphan", "ultra-rare"],
    "Reproductive Health": ["Reproductive Health", "Prenatal Care", "Feminine Health", "Women Health"],
    "Surgery": ["Surgery", "Surgical"],
    "Transplant":["Transplant"],
    "Urology": ["Urology"],


    "Biomarkers": ["Biomarkers"],
    "Diagnostics": ["diagnostic", "diagnostics", "monitoring"],
    "Educational/Training Materials": ["Educational Materials", "Training Materials", "Training and Education"],
    "Medical Devices": ["Medical Devices", "Devices", "Medical Technology", "Instruments","implant", "sensor"],
    "Medical Equipment": ["Medical Equipment", "equipment"],
    "Novel Targets" : ["Novel Targets"],
    "Research Tools" : ["Research Tools", "Helping Researchers", "Tools for Researchers"],
    "Animal Models": ["animal models"],
    "Antibody": ["antibody", "monoclonal"],
    "Antigen": ["Antigen"],
    "Assay": ["Assay"],
    "Bacterial Strain" :["bacterial strain", "bacteria use for research", "bacteria for research"],
    "Cell Line" : ["cell line", "cell lines"],
    "Plasmid/Vector": ["plasmid", "vector"],
    "Protein (Research Tool)": ["protein", "proteins"],
    "Software": ["software", "algorithm"],
    "Imaging Software": ["imaging", "imaging software", "radiology"],
    "Technology Platform/Enabling Technology": ["Technology Platform", "platform", "Enabling Technology", "Epic"],
    "Therapeutics" : ["Therapeutics", "Therapeutics Solutions", "Therapies", "Therapy"],
}
# Specific types of therapy
therapy_subtypes = {
    "ASOs": ["ASOs", "ASO", "Antisense oligonucleotide"],
    "Cell Therapy": ["cell therapy", "stem cell", "unicellular", "multicellular", "car t", "t cell"],
    "Gene Therapy": ["gene therapy", "genetic therapy"],
    "Large Molecule": ["large molecule", "large molecules"],
    "Microbiome":["microbiome","microbiotic","microbiomes"],
    "Nutraceuticals/Supplements": ["Nutraceuticals", "Supplements", "Nutritional Supplements"],
    "Peptide": ["Peptide","peptides"],
    "Protein": ["Protein", "proteins"],
    "RNA (ie. mRNA, siRNA)": ["rna", "mrna", "sirna"],
    "Small Molecule": ["small molecule", "small molecule therapy"],
}

# Flatten both maps
flat_keyword_map = {
    kw.lower(): cat # kw is indivudual keyword and cat is category
    for cat, kws in category_keyword_map.items() #iterate through categories and their lists
    for kw in kws # iterate through each keyword in list
}
flat_therapy_map = {
    kw.lower(): subtype
    for subtype, kws in therapy_subtypes.items()
    for kw in kws
}
# converts dicts to new dict where individual keyword from each list becomes key and category is value

# Matching logic
def match_keywords(text, general_map, therapy_map):
    text = text.lower()
    matched = set()
    keys = set()

    # General keyword matching
    for kw, cat in general_map.items():
        if kw in text:
            matched.add(cat)
            keys.add(kw)

    # maybe some might have therapy in text but not practice it; pass if no subtypes are matched
    # Therapy keyword logic
    if "therapy" in text or "therapeutics" in text:
        found_subtypes = [subcat for kw, subcat in therapy_map.items() if kw in text]
        if found_subtypes:
            matched.update(found_subtypes)
        else:
            matched.add("Therapeutics")  # fallback if no subtype found

    # instead of returning nan, maybe reiterate functions with new params
    # return ", ".join(sorted(matched)) if matched else np.nan
    if matched:
      matched_str = ", ".join(sorted(matched))
      keys_str = ", ".join(sorted(keys))
      if keys:
        return f"Categories: {matched_str}\nKeys: {keys_str}"
      else:
        return f"Categories: {matched_str}"
    else:
      return np.nan

# Apply categorization
df["Categories_Predicted"] = df["Scraped_Text"].apply(lambda text: match_keywords(text, flat_keyword_map, flat_therapy_map))

# Optional ML predictions if some categories are still missing
labeled = df.dropna(subset=["Categories_Predicted"])
if not labeled.empty:
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    X_train = vectorizer.fit_transform(labeled["Scraped_Text"])
    y_train = labeled["Categories_Predicted"]
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    unlabeled = df[df["Categories_Predicted"].isna()]
    X_test = vectorizer.transform(unlabeled["Scraped_Text"])
    y_pred = clf.predict(X_test)

    df.loc[unlabeled.index, "ML_Predicted"] = y_pred
else:
    df["ML_Predicted"] = np.nan

# Save Excel output
output_path = "Scraping_Results.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
    df.to_excel(writer, sheet_name="All Data", index=False)
    df[["Name", "Website", "Categories_Predicted", "ML_Predicted"]].to_excel(
        writer, sheet_name="Summary", index=False)
    df[df["Categories_Predicted"].isna()][["Name", "Website", "ML_Predicted"]].to_excel(
        writer, sheet_name="ML Fill-In", index=False)

# Download in Colab
files.download(output_path)
