import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
import re
from google.colab import files
from functools import partial

# Load Excel
df = pd.read_excel("Companies.xlsx")

# Prepend 'https://' to URLs if missing and validate basic URL structure
def clean_and_validate_url(url):
    if pd.isna(url) or not isinstance(url, str):
        return np.nan # Return NaN for non-string or NaN entries

    url = url.strip()

    # Check for clearly invalid URL patterns (e.g., phone numbers, "Not Found", empty strings)
    # This regex is a heuristic and might need adjustment for specific edge cases.
    # It looks for common non-URL patterns: starts with numbers/parentheses (like phone),
    # or common phrases like "Not Found", or if it's just whitespace.
    if re.fullmatch(r'^\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$', url) or \
       re.fullmatch(r'^\s*(not found|n/a|none|unknown)\s*$', url, re.IGNORECASE) or \
       url == '':
        print(f"Skipping clearly invalid URL pattern: {url}")
        return np.nan

    # Check if it contains spaces or other characters not allowed in domain names (simple check)
    if ' ' in url and not url.startswith('http'): # Allow spaces if it's already a full http/https URL (less common, but possible in malformed data)
        print(f"Skipping URL with spaces and no scheme: {url}")
        return np.nan

    # Prepend https:// if no scheme is present
    if not url.startswith('http://') and not url.startswith('https://'):
        # A basic check to see if it even resembles a domain before prepending
        if '.' in url and len(url) > 3: # min length like "a.co"
            return 'https://' + url
        else:
            print(f"Skipping non-domain-like string without scheme: {url}")
            return np.nan
    return url

df["Website"] = df["Website"].apply(clean_and_validate_url)
# Drop rows where 'Website' became NaN after validation
df.dropna(subset=["Website"], inplace=True)

# Clean text function
def clean_text(text):
    if isinstance(text, str):
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text) # replaces non-printable ASCII letters with a space
    return text

for col in df.columns:
    if df[col].dtype == 'object': # object data type indicates it contains strings or mixed types
        df[col] = df[col].apply(clean_text)

# Scraping function - now accepts selenium_driver as an argument
def scrape_relevant_sections(url, selenium_driver=None):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')

        keywords = ["about", "product", "pipeline", "technology", "mission", "vision", "platform", "solution"]
        relevant_texts = [] # List to store individual text sections
        keyword_match_found = False # Flag to track if any keyword sections were found

        # Search in h1, h2, h3, a, strong for keywords
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

        # Fallback 1: If no important sections are matched by keywords, use longer p tags
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
            relevant_texts.extend(long_paragraphs)

            # Fallback 2: If p tags didn't yield enough, try longer div tags
            if not long_paragraphs and not relevant_texts: # Only if p tags and initial search didn't add anything substantial
                long_div_texts = []
                div_tags = soup.find_all("div")
                for div_tag in div_tags:
                    if len(long_div_texts) >= 5: # Stop if we already found 5
                        break
                    div_text = div_tag.get_text(strip=True)
                    div_text_processed = re.sub(r'\s+', ' ', div_text).lower().strip()
                    if len(div_text_processed) > 50: # Using the same threshold for consistency
                       long_div_texts.append(div_text_processed)
                relevant_texts.extend(long_div_texts)

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
    "Allergy" : ["Allergy", "Allergies", "allergen", "IgE", "anaphylaxis", "urticaria", "hives", "epinephrine auto-injector", "allergic"],
    "Anesthesia": ["Anesthesia", "Anesthetics", "Anesthesiology", "intubation", "analgesia", "sedation", "ASA classification", "perioperative", "PACU"],
    "Autoimmune": ["autoimmune", "lupus", "rheumatoid","AIDS/HIV", "autoantibodies", "multiple sclerosis", "type 1 diabetes", "immunosuppression"],
    "Cardiology": ["cardiology", "heart", "cardiac", "arrhythmia", "CHD", "echocardiogram", "ECG/EKG", "pacemaker", "cardiomyopathy" ,"electrophysiology"],
    "Cardiovascular": ["Cardiovascular","vascular", "hypertension", "atherosclerosis", "stroke", "myocardial ischemia"],
    "Dermatology": ["dermatology", "skin", "eczema", "psoriasis", "acne", "dermatitis", "pruritus", "skin biopsy"],
    "Endocrinology": ["Endocrine", "Endocrinology", "adrenal", "thyroid", "hormones"],
    "Fetal/Newborn Medicine": ["Newborn", "Newborns", "Fetus", "Fetal", "neonatology", "prenatal diagnosis", "congentital"],
    "Gastroenterology": ["Gastroenterology", "Gastrointestinal", "celiac", "endoscopy", "Crohn's", "pancreatitis", "colonscopy"],
    "Hematology": ["Hematology", "anemia", "sickle cell", "hemophilia", "bone marrow", "coagulation", "thrombocytopenia"],
    "Immunology": ["Immunology", "Immune System", "immunoglobulins", "immune deficiency", "lymphocytes", "cytokines"],
    "Infectious Disease": ["infectious", "viral", "bacterial", "covid", "sars-cov-2", "antivirals", "sepsis", "vaccination"],
    "Metabolic": ["metabolic", "obesity", "diabetes", "insulin", "glucose", "hypoglysemia", "hyperammonemia"],
    "Nephrology": ["Nephrology","Kidney", "proteinuria", "hematuria", "nephrotic syndrome", "hypertension"],
    "Neurology": ["neurology", "brain", "neuron", "epilepsy", "alzheimer", "parkinsons", "Epilepsy","psychiatry","depression", "anxiety", "mental health", "seizures", "stroke", "neuropathy", "neuroimaging", "neuromuscular"],
    "Psychiatry": ["psychiatry", "depression", "anxiety", "mental health", "ADHD", "PTSD", "psychosis", "psychopharmacology"],
    "Neuroscience": ["Neuroscience", "brain circuit", "cognition", "neurodevelopment", "neuroplasticity", "neurogenetics"],
    "Obesity": ["Obesity", "Weight Loss", "Weight Management", "metabolic syndrome", "insulin resistance", "fatty liver"],
    "Oncology": ["oncology", "cancer", "tumor", "carcinoma", "chemotherapy", "immunotherapy", "leukemia", "lymphoma"],
    "Opthamology": ["Opthamology", "amblyopia", "strabismus", "glaucoma", "retina", "cornea", "fundus exam", "eye trauma"],
    "Orthopedics": ["Orthopedic", "sports injury", "ligament/ACL", "joint pain", "hip dysplasia", "physical therapy"],
    "Pulmonary": ["pulmonary", "respiratory", "lung", "asthma", "COPD", "cystic fibrosis", "pneumonia", "bronchiolitis"],
    "Rare Diseases": ["rare disease", "orphan", "ultra-rare", "diagnostic odyssey"],
    "Reproductive Health": ["Reproductive Health", "Prenatal Care", "Feminine Health", "Women Health", "sexual health", "PCOS", "STI screening"],
    "Surgery": ["Surgery", "Surgical", "laparoscopic"],
    "Transplant":["Transplant", "organ allocation", "HLA matching"],
    "Urology": ["Urology", "UTI", "hematuria", "kidney stones", "urodynamics"],


    "Biomarkers": ["Biomarkers", "surrogate endpoint", "ROC/AUC"],
    "Diagnostics": ["diagnostic", "diagnostics", "monitoring", "lab-developed test", "clinical ulity"],
    "Educational/Training Materials": ["Educational Materials", "Training Materials", "Training and Education", "instructional design", "curriculum"],
    "Medical Devices": ["Medical Devices", "Devices", "Medical Technology", "Instruments","implant", "sensor", "FDA 510(k)", "PMA"],
    "Medical Equipment": ["Medical Equipment", "equipment", "electrical safety"],
    "Research Tools" : ["Research Tools", "Helping Researchers", "Tools for Researchers", "automation", "reagents"],
    "Animal Models": ["animal models", "mouse model", "transgenic", "xenograft"],
    "Antibody": ["antibody", "monoclonal"],
    "Antigen": ["Antigen"],
    "Assay": ["Assay"],
    "Bacterial Strain" :["bacterial strain", "bacteria use for research", "bacteria for research"],
    "Cell Line" : ["cell line", "cell lines"],
    "Plasmid/Vector": ["plasmid", "vector"],
    "Protein (Research Tool)": ["protein", "proteins"],
    "Software": ["software", "algorithm"],
    "Imaging Software": ["imaging", "imaging software", "radiology"],
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
    if pd.isna(text) or not isinstance(text, str): # Handle NaN or non-string input from Scraped_Text
        return np.nan

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
        # else:
        #    matched.add("Therapeutics")  # fallback if no subtype found
    
    # companies that trigger both cancer and infectious diseases should just be oncology
    if "Oncology" in matched and "Infectious Disease" in matched:
      matched.remove("Infectious Disease")

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
