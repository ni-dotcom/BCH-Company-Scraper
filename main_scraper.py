import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import re
from google.colab import files, userdata
from playwright.async_api import async_playwright
import asyncio
from urllib.parse import urljoin, urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

# =========================
# CONFIG
# =========================
PAGE_HINT_TERMS = [
    "about", "product", "pipeline", "technology", "mission",
    "vision", "platform", "solution", "science", "company"
]

POSITIVE_LINK_TERMS = [
    "about", "company", "who-we-are", "our-story", "mission",
    "technology", "platform", "pipeline", "science", "products",
    "solutions", "research"
]

NEGATIVE_LINK_TERMS = [
    "careers", "jobs", "news", "press", "blog", "events",
    "privacy", "terms", "legal", "contact", "support",
    "login", "investor", "cookie"
]

PLAYWRIGHT_SEMAPHORE = asyncio.Semaphore(5)  # check with higher number

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


# =========================
# LOAD EXCEL
# =========================
df = pd.read_excel("Companies_test.xlsx")


# =========================
# URL CLEANING
# =========================
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
        return 'Check website: ' + url

    # Check if it contains spaces or other characters not allowed in domain names (simple check)
    if ' ' in url and not url.startswith('http'): # Allow spaces if it's already a full http/https URL (less common, but possible in malformed data)
        print(f"Skipping URL with spaces and no scheme: {url}")
        return 'Check website: ' + url

    # Prepend https:// if no scheme is present
    if not url.startswith('http://') and not url.startswith('https://'):
        # A basic check to see if it even resembles a domain before prepending
        if '.' in url and len(url) > 3: # min length like "a.co"
            return 'https://' + url
        else:
            print(f"Skipping non-domain-like string without scheme: {url}")
            return 'Check website: ' + url
    return url

df["Website"] = df["Website"].apply(clean_and_validate_url)
check_website = df[df["Website"].str.contains('Check website: ', na=True)][["Name", "Website"]]
# Drop rows with invalid urls
df = df[~df["Website"].str.contains('Check website: ', na=True)].copy()


# =========================
# TEXT CLEANING
# =========================
def clean_text(text):
    if isinstance(text, str):
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text) # replaces non-printable ASCII letters with a space
    return text

for col in df.columns:
    if df[col].dtype == 'object': # object data type indicates it contains strings or mixed types
        df[col] = df[col].apply(clean_text)


# =========================
# SCRAPING HELPERS
# =========================
def unique_join(items, sep="; "):
    seen = []
    for item in items:
        if item and item not in seen:
            seen.append(item)
    return sep.join(seen)

# check meta descriptions in a website for more info
def extract_meta_descriptions(soup):
    meta_texts = []

    for attrs in [
        {"name": "description"},
        {"property": "og:description"},
        {"name": "twitter:description"}
    ]:
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            text = re.sub(r"\s+", " ", tag["content"]).lower().strip()
            if len(text) > 30:
                meta_texts.append(text)

    return list(dict.fromkeys(meta_texts))

# find other useful links on the homepage
def extract_candidate_links(base_url, soup, max_links=3):
    base_domain = urlparse(base_url).netloc.replace("www.", "")
    candidates = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        anchor_text = a.get_text(" ", strip=True).lower()

        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue

        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        link_domain = parsed.netloc.replace("www.", "")

        if link_domain and link_domain != base_domain:
            continue

        combined = (anchor_text + " " + full_url.lower()).strip()

        if any(term in combined for term in NEGATIVE_LINK_TERMS):
            continue

        score = sum(term in combined for term in POSITIVE_LINK_TERMS)
        if score > 0 and full_url.rstrip("/") != base_url.rstrip("/"):
            candidates.append((score, full_url))

    candidates.sort(key=lambda x: x[0], reverse=True)

    seen = set()
    final_links = []
    for score, link in candidates:
        if link not in seen:
            seen.add(link)
            final_links.append(link)
        if len(final_links) >= max_links:
            break

    return final_links

# search a webpage for possible useful text
def extract_relevant_text_from_soup(soup):
    for tag in soup(["script", "style", "noscript", "nav", "footer"]):
        tag.decompose()

    relevant_texts = []
    keyword_section_found = False
    fallback_used = False

    # meta description often helps
    meta_texts = extract_meta_descriptions(soup)
    relevant_texts.extend(meta_texts)

    # heading / section-based extraction
    for tag in soup.find_all(["h1", "h2", "h3", "a", "strong"]):
        tag_text = tag.get_text(" ", strip=True).lower()
        if any(term in tag_text for term in PAGE_HINT_TERMS):
            parent = tag.find_parent()
            if parent and parent.name not in ["nav", "footer", "header"]:
                section_text = parent.get_text(separator=" ", strip=True)
                section_text = re.sub(r"\s+", " ", section_text).lower().strip()
                if len(section_text) > 50:
                    relevant_texts.append(section_text)
                    keyword_section_found = True

    # Fallback 1: If no important sections are matched by keywords, use longer p tags
    if not keyword_section_found:
        long_paragraphs = []
        paragraphs = soup.find_all("p")
        for p in paragraphs:
            if len(long_paragraphs) >= 5: # Stop if we already found 5
                break
            para_text = p.get_text(strip=True)
            para_text_processed = re.sub(r'\s+', ' ', para_text).lower().strip()
            if len(para_text_processed) > 50:
                long_paragraphs.append(para_text_processed)
                fallback_used = True
        relevant_texts.extend(long_paragraphs)

        # Fallback 2: If p tags didn't yield enough, try longer div tags
        if not long_paragraphs:
            long_div_texts = []
            div_tags = soup.find_all("div")
            for div_tag in div_tags:
                if len(long_div_texts) >= 5: # Stop if we already found 5
                    break
                div_text = div_tag.get_text(strip=True)
                div_text_processed = re.sub(r'\s+', ' ', div_text).lower().strip()
                if len(div_text_processed) > 50: # Using the same threshold for consistency
                    long_div_texts.append(div_text_processed)
                    fallback_used = True
            relevant_texts.extend(long_div_texts)

    relevant_texts = list(dict.fromkeys(relevant_texts))  # De-duplicate repeated sections
    combined = "\n".join(relevant_texts) # Construct the combined string, joining with newlines

    method_used = "empty"
    if keyword_section_found:
        method_used = "keyword_section"
    elif fallback_used:
        method_used = "fallback"
    if meta_texts:
        method_used = method_used + " with meta"
    return combined, method_used

# =========================
# REQUESTS SCRAPER
# =========================
def scrape_page_with_requests(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = session.get(url, headers=headers, timeout=8, allow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        text, detail = extract_relevant_text_from_soup(soup)

        return text, soup, f"requests_{detail}", url

    except Exception as e:
        print(f"Requests error for {url}: {e}")
        return "", None, "requests_error", url


# =========================
# PLAYWRIGHT SCRAPER FALLBACK
# =========================
async def scrape_with_playwright_page(page, url):
    try:
        await page.goto(url, timeout=12000, wait_until="domcontentloaded")
        await page.wait_for_timeout(1500)

        html = await page.content()
        soup = BeautifulSoup(html, "html.parser")
        text, detail = extract_relevant_text_from_soup(soup)
        return text, f"playwright_{detail}"

    except Exception as e:
        print(f"Playwright failed for {url}: {e}")
        return "", "playwright_error"

# =========================
# FIRST SCRAPE PASS
# =========================
async def scrape_site_requests_only(url):
    return await asyncio.to_thread(scrape_page_with_requests, url)

async def first_pass_requests(df, batch_size=25):
    scraped_texts = []
    scrape_methods = []
    source_urls = []

    urls = df["Website"].tolist()

    # run functions on a batch of urls at once to save time
    for start in range(0, len(urls), batch_size):
        batch = urls[start:start + batch_size]
        tasks = [scrape_site_requests_only(url) for url in batch]
        results = await asyncio.gather(*tasks)

        for text, soup, method, source in results:
            scraped_texts.append(text)
            scrape_methods.append(method)
            source_urls.append(source)

    df["Scraped_Text"] = scraped_texts
    df["Scrape_Method"] = scrape_methods
    df["Source_URL"] = source_urls
    return df

df = await first_pass_requests(df, batch_size=25)


# =======================
# KEYWORD MAPS
# =======================
category_keyword_map = {
    "Allergy" : ["Allergy", "Allergies", "allergen", " IgE ", "anaphylaxis", "urticaria", "hives", "epinephrine auto-injector", "allergic"],
    "Anesthesia": ["Anesthesia", "Anesthetics", "Anesthesiology", "intubation", "analgesia", "sedation", "ASA classification", "perioperative", " PACU "],
    "Autoimmune": ["autoimmune", "lupus", "rheumatoid", "autoantibodies", "multiple sclerosis"],
    "Cardiology": ["cardiology", "arrhythmia", "echocardiogram", "ECG/EKG", "pacemaker", "cardiomyopathy" ,"electrophysiology"],
    "Cardiovascular": ["Cardiovascular", "atherosclerosis", "myocardial ischemia"],
    "Dermatology": ["dermatology", "eczema", "psoriasis", "acne", "dermatitis", "pruritus", "skin biopsy"],
    "Endocrinology": ["Endocrine", "Endocrinology", "adrenal", "thyroid"],
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
    "Opthamology": ["Opthamology", "amblyopia", "strabismus", "glaucoma", "retina", "cornea", "fundus exam", "eye trauma"],
    "Orthopedics": ["Orthopedic", "sports injury", "ligament/ACL", "joint pain", "hip dysplasia", "physical therapy"],
    "Pulmonary": ["pulmonary", "asthma", "COPD", "cystic fibrosis", "bronchiolitis"],
    "Rare Diseases": ["rare disease", "orphan", "ultra-rare", "diagnostic odyssey"],
    "Reproductive Health": ["Reproductive Health", "Feminine Health", "PCOS"],
    "Surgery": ["laparoscopic"],
    "Transplant":["Transplant", "organ allocation", "HLA matching"],
    "Urology": ["Urology", " UTI ", "kidney stones", "urodynamics"],


    "Biomarkers": ["Biomarkers", "surrogate endpoint", "ROC/AUC"],
    "Diagnostics": ["diagnostic", "diagnostics", "monitoring", "lab-developed test", "clinical utility"],
    "Educational/Training Materials": ["Educational Materials", "Training Materials", "Training and Education", "instructional design", "curriculum"],
    "Medical Devices": ["Medical Devices", "Medical Technology", "implant", "FDA 510(k)", " PMA "],
    "Medical Equipment": ["Medical Equipment", "electrical safety"],
    "Research Tools" : ["Research Tools", "Helping Researchers", "Tools for Researchers"],
    "Animal Models": ["animal models", "mouse model", "transgenic", "xenograft"],
    "Antibody": ["polyclonal", "lot-to-lot variability", "cross-reactivity"],
    "Antigen": ["Antigen", "immunogen", "hapten", "immunogenicity", "pathogen-associated"],
    "Assay": [" PCR ", "immunoassay", "limit of detection"],
    "Bacterial Strain" :["bacterial strain", "bacteria use for research", "bacteria for research", "culture conditions", "reference strain"],
    "Cell Line" : ["cell line", "cell lines", "CRISPR editing", "phenotype drift"],
    "Plasmid/Vector": ["plasmid", "antibiotic resistance gene", "cloning sites", "lentiviral", "titer"],
    "Protein (Research Tool)": ["purification", "activity assay", "binding kinetics", "post-translational modifications"],
    "Software": ["interoperability"],
    "Imaging Software": ["imaging software", " DICOM", " PACS "],
}
# Specific types of therapy
therapy_subtypes = {
    "ASOs": ["ASOs", "ASO", "Antisense oligonucleotide", "RNA splicing", "exon skipping", "RNase H", "gapmer", "GalNAc", "intrathecal dosing"],
    "Cell Therapy": ["cell therapy", "stem cell", "unicellular", "multicellular", "car t", "CAR-T", "TCR-T"],
    "Gene Therapy": ["gene therapy", "genetic therapy", "lentiviral vector", "insertional mutagenesis", " AAV "],
    "Large Molecule": ["large molecule", "large molecules", "Fc region", "glycosylation", "half-life extension", "cold chain"],
    "Microbiome":["microbiome","microbiotic","microbiomes", "dysbiosis", "16S", "shotgun metagenomics", "SCFAs", "probiotics", "prebiotics", "fecal microbiota transplant", "colonization resistance", "antibiotics effect", "microbial ecology"],
    "Nutraceuticals/Supplements": ["Nutraceuticals", "Supplements", "Nutritional Supplements", "FDA DSHEA", "evidence grading", "USP verification"],
    "Peptide": ["Peptide","peptides", "proteolysis", "cyclization", "subcutaneous injection", "receptor antagonist"],
    "Protein": ["Protein", "proteins", "enzyme replacement", "PK clearance", "post-translational modifications"],
    "RNA (ie. mRNA, siRNA)": ["mrna", "sirna", "lipid nanoparticles", "innate immune activation", " RISC ", "tissue targeting"],
    "Small Molecule": ["small molecule", "small molecule therapy"],
}

# Flatten both maps
flat_keyword_map = {
    kw.lower(): cat
    for cat, kws in category_keyword_map.items() 
    for kw in kws
}
# converts dicts to new dict where individual keyword from each list becomes key and category is value
flat_therapy_map = {
    kw.lower(): subtype
    for subtype, kws in therapy_subtypes.items()
    for kw in kws
}

# =========================
# MATCHING
# =========================
def keyword_found(text, kw):  # prevent false positives in keyword-matching (eliminated 8)
    pattern = r'\b' + re.escape(kw.lower().strip()) + r'\b'
    return re.search(pattern, text.lower()) is not None

# Matching logic
def match_keywords(text, general_map, therapy_map):
    if pd.isna(text) or not isinstance(text, str):
        return np.nan, np.nan, np.nan

    text = text.lower()
    matched = []
    trusty_cat = set()
    maybe_cat = set()
    keys = set()

    # general keyword matching
    for kw, cat in general_map.items():
        if keyword_found(text, kw):
            matched.append(cat)
            keys.add(kw)

    # therapy matching
    for kw, subcat in therapy_map.items():
        if keyword_found(text, kw):
            matched.append(subcat)
            keys.add(kw)

    # separate categories that show up more than once and are more trustable
    for cat in matched:
      if cat in maybe_cat or cat in trusty_cat:
        trusty_cat.add(cat)
        maybe_cat.discard(cat)
      else:
        maybe_cat.add(cat)

    # companies that trigger both cancer and infectious diseases should just be oncology
    if "Oncology" in matched and "Infectious Disease" in matched:
        matched.remove("Infectious Disease")

    categories_str = ", ".join(sorted(trusty_cat)) if trusty_cat else np.nan
    keys_str = ", ".join(sorted(keys)) if keys else np.nan
    maybe_categories = ", ".join(sorted(maybe_cat)) if maybe_cat else np.nan

    return categories_str, keys_str, maybe_categories

# =========================
# FIRST CATEGORIZATION PASS
# =========================
df[["Categories_Predicted", "Matched_Keywords", "Possible_More_Categories"]] = df["Scraped_Text"].apply(
    lambda text: pd.Series(match_keywords(text, flat_keyword_map, flat_therapy_map)))

# =========================
# OPTIONAL SECOND PASS:
# only re-scrape uncategorized rows
# =========================
async def second_pass_uncategorized(df):
    mask = df["Categories_Predicted"].isna() & df["Website"].notna()
    retry_indices = df[mask].index.tolist()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        for i, idx in enumerate(retry_indices, start=1):
            url = df.at[idx, "Website"]

            collected_texts = []
            methods = []
            source_urls = []

            # 1) try subpages with requests
            homepage_text, homepage_soup, homepage_method, homepage_source = scrape_page_with_requests(url)
            collected_texts.append(homepage_text)
            methods.append(homepage_method)
            source_urls.append(homepage_source)

            # 2) internal links with playwright
            if homepage_soup is not None:
                candidate_links = extract_candidate_links(url, homepage_soup, max_links=3)

                for link in candidate_links:
                    sub_text, _, sub_method, sub_source = scrape_page_with_requests(link)

                    if not sub_text or len(sub_text) < 100:
                        page = await browser.new_page(user_agent="Mozilla/5.0")
                        pw_text, pw_method = await scrape_with_playwright_page(page, link)
                        await page.close()

                        if pw_text and len(pw_text) > 100:
                            sub_text = pw_text
                            sub_method = pw_method

                    if sub_text:
                        collected_texts.append(f"[source: {link}]\n{sub_text}")
                        methods.append(sub_method)
                        source_urls.append(sub_source)

            # 3) only if still weak, try playwright on homepage
            total_text = "\n\n".join(collected_texts).strip()
            if (not total_text or len(total_text) < 150):
                page = await browser.new_page(user_agent="Mozilla/5.0")
                pw_text, pw_method = await scrape_with_playwright_page(page, url)
                await page.close()

                if pw_text:
                    collected_texts.append(f"[source: {url}]\n{pw_text}")
                    methods.append(pw_method)
                    source_urls.append(url)

            # dedupe
            collected_texts = list(dict.fromkeys([t for t in collected_texts if t]))
            methods = list(dict.fromkeys([m for m in methods if m]))
            source_urls = list(dict.fromkeys([s for s in source_urls if s]))

            final_text = "\n\n".join(collected_texts).strip()

            if final_text:
                df.at[idx, "Scraped_Text"] = final_text
                df.at[idx, "Scrape_Method"] = f"second_pass:{unique_join(methods)}"
                df.at[idx, "Source_URL"] = unique_join(source_urls)

                cats, kws, more_cats = match_keywords(final_text, flat_keyword_map, flat_therapy_map)
                df.loc[idx, ["Categories_Predicted", "Matched_Keywords", "Possible_More_Categories"]] = [cats, kws, more_cats]

        await browser.close()

    return df

df = await second_pass_uncategorized(df)

# =========================
# REMOVE SECURITY-TRIGGERING WEBSITES
# =========================
# So ML model doesn't get confused
protected = df[df["Scraped_Text"].str.contains('security service', na=False)][["Name", "Website", "Primary Key", "Scraped_Text"]]
df = df[~df["Scraped_Text"].str.contains('security service', na=False)].copy()

# =========================
# OPTIONAL ML FILL-IN
# =========================
def split_categories(cat_string):
    if pd.isna(cat_string) or not isinstance(cat_string, str) or not cat_string.strip():
        return []
    return [c.strip() for c in cat_string.split(",") if c.strip()]

def ml_prediction(df):
    # Read already scraped excel
    labeled = pd.read_excel("CompaniesExport_refined.xlsx", sheet_name="All Data")

    labeled["Scraped_Text"] = (
      labeled["Company type(s)"].fillna("").astype(str) + "\n" +
      labeled["Brief description"].fillna("").astype(str) + "\n" +
      labeled["Description"].fillna("").astype(str) + "\n" +
      labeled["Primary therapeutic area(s)"].fillna("").astype(str) + "\n" +
      labeled["Secondary therapeutic area(s)"].fillna("").astype(str) + "\n" +
      labeled["Partnering objectives"].fillna("").astype(str) + "\n" +
      labeled["Company objectives"].fillna("").astype(str)
    )

    # Only try ML on rows with text but no rule-based categories
    unlabeled = df[
        df["Categories_Predicted"].isna() &
        df["Scraped_Text"].notna() &
        (df["Scraped_Text"].str.strip() != "")
    ].copy()

    if not labeled.empty and not unlabeled.empty:
        labeled["Category_List"] = labeled["Final_Categories"].apply(split_categories)

        # Remove rows with no parsed categories
        labeled = labeled[labeled["Category_List"].map(len) > 0].copy()

        if not labeled.empty:
            vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2), min_df=2)

            X_train = vectorizer.fit_transform(labeled["Scraped_Text"])

            mlb = MultiLabelBinarizer()
            y_train = mlb.fit_transform(labeled["Category_List"])

            clf = OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced"))

            clf.fit(X_train, y_train)

            X_test = vectorizer.transform(unlabeled["Scraped_Text"])

            # Probability scores per category
            y_prob = clf.predict_proba(X_test)

            # Threshold for assigning a category
            threshold = 0.50

            ml_pred_categories = []
            ml_confidence = []

            for probs in y_prob:
                selected = [mlb.classes_[i] for i, p in enumerate(probs) if p >= threshold]

                # If nothing passes threshold, leave blank
                if not selected:
                    ml_pred_categories.append(np.nan)
                    ml_confidence.append(float(np.max(probs)))
                else:
                    ml_pred_categories.append(", ".join(sorted(selected)))
                    ml_confidence.append(float(np.max(probs)))

            df["ML_Predicted"] = np.nan
            df["ML_Confidence"] = np.nan

            df.loc[unlabeled.index, "ML_Predicted"] = ml_pred_categories
            df.loc[unlabeled.index, "ML_Confidence"] = ml_confidence
        else:
            df["ML_Predicted"] = np.nan
            df["ML_Confidence"] = np.nan
    else:
        df["ML_Predicted"] = np.nan
        df["ML_Confidence"] = np.nan

    return df

df = ml_prediction(df)

# =========================
# SAVE OUTPUT
# =========================
uncategorized = df[df["Categories_Predicted"].isna() & df["Possible_More_Categories"].isna() & df["ML_Predicted"].isna()].copy()

output_path = "Scraping_Results.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
    df.filter(items=["Name", "Scraped_Text", "Website", "Source_URL", "Categories", "Categories_Predicted", "Matched_Keywords", "Possible_More_Categories", "ML_Predicted", "ML_Confidence"]).to_excel(
        writer, sheet_name="Summary", index=False)
    df[["Name", "Primary Key", "Website", "Categories_Predicted", "Possible_More_Categories"]].to_excel(
        writer, sheet_name="Categories Scraped", index=False)
    df[df["Categories_Predicted"].isna()][["Name", "Primary Key", "Website", "Scraped_Text", "ML_Predicted", "ML_Confidence"]].to_excel(
        writer, sheet_name="ML Fill-In", index=False)
    uncategorized[["Name", "Primary Key", "Website", "Scraped_Text"]].to_excel(
        writer, sheet_name="Uncategorized", index=False)
    protected.to_excel(writer, sheet_name="Protected Websites", index=False)
    df.to_excel(writer, sheet_name="All Data", index=False)

    check_website.to_excel(writer, sheet_name="Check_websites", index=False)

# Download in Colab
files.download(output_path)
