import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import re
from google.colab import files, userdata
from time import sleep
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from playwright.async_api import async_playwright
import asyncio
from urllib.parse import urljoin, urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import unicodedata
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

# =========================
# LOAD EXCEL
# =========================
df = pd.read_excel("Companies_NeedsCategories_batch2.xlsx")

# =========================
# CONFIG
# =========================
PAGE_HINT_TERMS = [
    "about", "product", "pipeline", "technology", "mission",
    "vision", "platform", "solution", "science", "company"
]

# for subpage navigation
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
NEWS_CATEGORY_TERMS = ["news", "press-release", "press", "blog", "media", "in-the-news"]

# for biography sections
TITLE_ABBR = r'\b(ph\.?d|m\.?d|mba|b\.?s|m\.?s|j\.?d|rn|dds|researcher|scientist|director|officer|president|professor|physician|executive|founder|chairman|chairwoman|chairperson)\b'
PRONOUNS = r'\b(he|she|his|her|him|they|their)\b'
BIO_PATTERNS = [
    r'\bis (a|an|the) (leading |senior |lead |principal |chief |founding )?[\w\- ]{0,30}(researcher|scientist|director|officer|president|professor|physician|executive|founder|chairman|chairwoman|chairperson)\b',
    r'\bholds (a|an) (ph\.?d|m\.?d|mba|b\.?s|j\.?d)\b',
    r'\breceived (his|her|their) (ph\.?d|m\.?d|mba|bachelor|master|degree)\b',
    r'\b(has over|with over|brings over) \d+ years? of experience\b',
    r'\bserves (as|on) the\b',
    r'\bbefore (joining|founding|co-founding)\b',
    r'\b(joined|prior to joining)\b.{0,60}\bas (a |an |the )?[A-Z][\w\- ]{2,40}\b',  # "MBA from Stanford"
    # honorific + name at sentence start
    r'\b(dr|prof|professor)\.?\s+[a-z]+\s+[a-z]+\b',
    # "is chief/head/director of X" — no article
    r'\bis (chief|head|director|vp|vice president|founder|co-founder)\s+of\b',
    # comma-set-off role appositive — ", company's founding CSO,"
    r",\s*[\w .&'-]{2,40}'s\s+(founding |former |current |chief )?[\w\- ]{2,30},",
    # broader profession/identity nouns beyond corporate titles
    r'\bis (a|an) [\w\-/]*(biochemist|biophysicist|immunologist|geneticist|neuroscientist|oncologist|cardiologist|epidemiologist|pharmacologist|biologist|chemist|engineer|physicist)\b',
]

PLAYWRIGHT_SEMAPHORE = asyncio.Semaphore(5)

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
# INITIAL CLEANING
# =========================
# clean URLs
def clean_and_validate_url(url):
    if pd.isna(url) or not isinstance(url, str):
        return np.nan # Return NaN for non-string or NaN entries

    url = url.strip()

    # Check for clearly invalid URL patterns (e.g., phone numbers, "Not Found", empty strings)
    # This regex is a heuristic and might need adjustment for specific edge cases.
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

# Drop rows with invalid urls
check_website = df[df["Website"].str.contains('Check website: ', na=True)][["Name", "Primary Key", "Website"]]
df = df[~df["Website"].str.contains('Check website: ', na=True)].copy()

# drop rows that shouldn't be categorized (universities, hospitals)
not_for_categorizing = df[df["Name"].str.contains("university|institute|hospital|foundation", case=False, na=True)][["Name", "Primary Key", "Website"]]
df = df[~df["Name"].str.contains("university|institute|hospital|foundation", case=False, na=True)].copy()

# =========================
# TEXT CLEANING
# =========================
def clean_text_for_excel(text):
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize("NFKC", text)
    text = ILLEGAL_CHARACTERS_RE.sub("", text)
    text = re.sub(r'[\x7F-\x9F\uD800-\uDFFF\uFFFE\uFFFF]', '', text)
    return text

# =========================
# SCRAPING HELPERS
# =========================
def unique_join(items, sep="; "):
    seen = []
    for item in items:
        if item and item not in seen:
            seen.append(item)
    return sep.join(seen)

# ignore sections that tell people's bios
def looks_like_bio(text):
    text_l = text.lower()

    pattern_hits = sum(bool(re.search(p, text_l)) for p in BIO_PATTERNS)
    # bios have a lot of pronouns and titles
    pronoun_hits = len(re.findall(PRONOUNS, text_l))
    title_hits = len(re.findall(TITLE_ABBR, text_l))

    return pattern_hits >= 1 or (pronoun_hits >= 4 and title_hits >= 1)

def is_news_like_node(tag):
    if tag is None or not tag.get("class"):
        return False

    classes = [str(c).lower() for c in tag.get("class", [])]

    # exact-token match: WordPress marks individual blog/news entries this way
    if "type-post" in classes:
        return True

    # exact-token match on category-* classes, e.g. category-in-the-news, category-press-releases
    for c in classes:
        if c.startswith("category-"):
            cat_value = c[len("category-"):]
            if any(term in cat_value for term in NEWS_CATEGORY_TERMS):
                return True

    return False

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
def extract_candidate_links(base_url, soup, max_links=6):
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
    for tag in soup.find_all(["article", "div", "li"]):
        if tag.decomposed:
            continue
        if is_news_like_node(tag):
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
        if any(term in tag_text for term in PAGE_HINT_TERMS) and "news" not in tag_text:
            parent = tag.find_parent()
            if parent and parent.name not in ["nav", "footer", "header"]:
                section_text = parent.get_text(separator=" ", strip=True)
                section_text = re.sub(r"\s+", " ", section_text).lower().strip()
                if len(section_text) > 50 and not looks_like_bio(section_text):
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
            if len(para_text_processed) > 50 and not looks_like_bio(para_text_processed):
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
                if len(div_text_processed) > 50 and not looks_like_bio(div_text_processed):
                    long_div_texts.append(div_text_processed)
                    fallback_used = True
            relevant_texts.extend(long_div_texts)

    relevant_texts = list(dict.fromkeys(relevant_texts))  # De-duplicate repeated sections
    combined = "\n".join(relevant_texts) # Construct the combined string, joining with newlines
    combined = clean_text_for_excel(combined)

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

        candidate_links = extract_candidate_links(url, soup, max_links=6) if soup is not None else []
        return text, soup, f"requests_{detail}", url, candidate_links

    except Exception as e:
        print(f"Requests error for {url}: {e}")
        return "", None, "requests_error", url, []

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
        return text, soup, f"playwright_{detail}", url

    except Exception as e:
        print(f"Playwright failed for {url}: {e}")
        return "", None, "playwright_error", url

# =========================
# COMBINED SCRAPER
# =========================
async def scrape_site(browser, url, first_pass=True):
    collected_texts = []
    methods = []
    source_urls = []

    # requests first
    text, soup, method, source, candidate_links = await asyncio.to_thread(scrape_page_with_requests, url)

    if text:
        collected_texts.append(text)
    methods.append(method)
    source_urls.append(source)

    # playwright fallback for homepage if weak
    if not text or len(text) < 50:
        async with PLAYWRIGHT_SEMAPHORE:  #
            page = await browser.new_page(user_agent="Mozilla/5.0")
            try:
                pw_text, pw_soup, pw_method, pw_source = await scrape_with_playwright_page(page, url)
            finally:
                await page.close()

        if pw_text and len(pw_text) > len(text):
            text = pw_text
            soup = pw_soup
            methods.append(pw_method)
            source_urls.append(pw_source)
            if collected_texts:
                collected_texts[0] = pw_text
            else:
                collected_texts.append(pw_text)

        # if requests had no soup, use playwright soup to get candidate links
        if soup is None and pw_soup is not None:
            soup = pw_soup
            candidate_links = extract_candidate_links(url, soup, max_links=6)

    # internal links
    if candidate_links:
        for link in candidate_links[:3] if first_pass else candidate_links[3:]: # scrape first 3 urls if first pass, otherwise last 3
            sub_text, _, sub_method, sub_source, _ = await asyncio.to_thread(scrape_page_with_requests, link)

            if not sub_text or len(sub_text) < 100:
                async with PLAYWRIGHT_SEMAPHORE:
                    page = await browser.new_page(user_agent="Mozilla/5.0")
                    try:
                        pw_sub_text, _, pw_sub_method, pw_sub_source = await scrape_with_playwright_page(page, link)
                    finally:
                        await page.close()

                if pw_sub_text and len(pw_sub_text) > len(sub_text):
                    sub_text = pw_sub_text
                    sub_method = pw_sub_method
                    sub_source = pw_sub_source

            if sub_text:
                collected_texts.append(f"[source: {link}]\n{sub_text}")
                methods.append(sub_method)
                source_urls.append(sub_source)

    # dedupe
    collected_texts = list(dict.fromkeys([t for t in collected_texts if t]))
    methods = list(dict.fromkeys([m for m in methods if m]))
    source_urls = list(dict.fromkeys([s for s in source_urls if s]))

    final_text = "\n\n".join(collected_texts).strip()
    final_method = unique_join(methods)
    final_sources = unique_join(source_urls)

    # language check
    if final_text:
        try:
            if detect(final_text[:2000]) != "en":
                return "", final_method + "; check website language", final_sources
        except LangDetectException:
            pass

    return final_text, final_method, final_sources

# =========================
# FIRST SCRAPE PASS
# =========================
async def first_pass_requests(df, batch_size=25):
    scraped_texts = []
    scrape_methods = []
    source_urls = []

    urls = df["Website"].tolist()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        for start in range(0, len(urls), batch_size):
            batch = urls[start:start + batch_size]
            tasks = [scrape_site(browser, url) for url in batch]
            results = await asyncio.gather(*tasks)

            for text, method, sources in results:
                scraped_texts.append(text)
                scrape_methods.append(method)
                source_urls.append(sources)

        await browser.close()

    df["Scraped_Text"] = scraped_texts
    df["Scrape_Method"] = scrape_methods
    df["URLs scraped"] = source_urls
    return df

df = await first_pass_requests(df, batch_size=25)


# =======================
# KEYWORD MAPS
# =======================
category_keyword_map = {
    "Allergy" : ["Allergy", "Allergies", "allergen", " IgE ", "anaphylaxis", "urticaria", "hives", "epinephrine auto-injector", "allergic"],
    "Anesthesia": ["Anesthesia", "Anesthetics", "Anesthesiology", "intubation", "analgesia", "sedation", "ASA classification", "perioperative", " PACU "],
    "Animal Health": ["Animal health", "animal illness", "vetenarians", "animal care", "pets"],
    "Autoimmune": ["autoimmune", "lupus", "rheumatoid", "autoantibodies", "multiple sclerosis"],
    "Cardiology": ["cardiology", "arrhythmia", "echocardiogram", "ECG/EKG", "pacemaker", "cardiomyopathy" ,"electrophysiology"],
    "Cardiovascular": ["Cardiovascular", "atherosclerosis", "myocardial ischemia"],
    "Dentistry": ["dentistry"],  # more?
    "Dermatology": ["dermatology", "eczema", "psoriasis", "acne", "dermatitis", "pruritus", "skin biopsy"],
    "Endocrinology": ["Endocrine", "Endocrinology", "adrenal", "thyroid"],
    "Epilepsy": ["epilepsy"],
    "Fetal/Newborn Medicine": ["Newborn", "Newborns", "Fetus", "Fetal", "neonatology", "prenatal diagnosis", "congenital"],
    "Gastroenterology": ["Gastroenterology", "Gastrointestinal", "celiac", "endoscopy", "Crohn's", "pancreatitis", "colonoscopy"],
    "Hematology": ["Hematology", "anemia", "sickle cell", "hemophilia", "thrombocytopenia"],
    "Immunology": ["Immunology", "immune deficiency"],
    "Infectious Disease": ["infectious", "covid", "sars-cov-2", "antivirals", "sepsis", " HIV "],
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
    "Regenerative Medicine": ["regenerative medicine", "xenotransplant", "tissue engineering", "organogenesis"],
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
            count = text.count(kw)
            matched.extend([cat] * count)
            keys.add(kw)

    # therapy matching
    for kw, subcat in therapy_map.items():
        if keyword_found(text, kw):
            count = text.count(kw)
            matched.extend([subcat] * count)
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
        trusty_cat.discard("Infectious Disease")

    categories_str = ", ".join(sorted(trusty_cat)) if trusty_cat else np.nan
    keys_str = ", ".join(sorted(keys)) if keys else np.nan
    maybe_categories = ", ".join(sorted(maybe_cat)) if maybe_cat else np.nan

    return categories_str, keys_str, maybe_categories

# =========================
# FIRST CATEGORIZATION PASS
# =========================
df[["Categories (predicted)", "Matched_Keywords", "More Categories (less certain)"]] = df["Scraped_Text"].apply(
    lambda text: pd.Series(match_keywords(text, flat_keyword_map, flat_therapy_map)))

# =========================
# OPTIONAL SECOND PASS:
# only re-scrape uncategorized rows
# =========================
async def second_pass_uncategorized(df):
    mask = (
        df["Categories (predicted)"].isna() &
        df["Website"].notna() &
        (~df["Scrape_Method"].str.contains("check website language", na=False))
    )
    retry_indices = df[mask].index.tolist()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        for idx in retry_indices:
            url = df.at[idx, "Website"]
            text, method, sources = await scrape_site(browser, url, first_pass=False)

            if text:
                df.at[idx, "Scraped_Text"] = text
                df.at[idx, "Scrape_Method"] = f"second_pass:{method}"
                df.at[idx, "URLs scraped"] = sources

                cats, kws, more_cats = match_keywords(text, flat_keyword_map, flat_therapy_map)
                df.loc[idx, ["Categories (predicted)", "Matched_Keywords", "More Categories (less certain)"]] = [cats, kws, more_cats]

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
        df["Categories (predicted)"].isna() &
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
            threshold = 0.65

            ml_pred_categories = []
            ml_confidence = []

            for probs in y_prob:
                selected = [mlb.classes_[i] for i, p in enumerate(probs) if p >= threshold]

                # If nothing passes threshold, leave blank
                if not selected:
                    ml_pred_categories.append(np.nan)
                    ml_confidence.append(np.nan)
                else:
                    ml_pred_categories.append(", ".join(sorted(selected)))
                    ml_confidence.append(float(np.max(probs)))

            df["ML Predicted"] = np.nan
            df["ML Confidence"] = np.nan

            df.loc[unlabeled.index, "ML Predicted"] = ml_pred_categories
            df.loc[unlabeled.index, "ML Confidence"] = ml_confidence
        else:
            df["ML Predicted"] = np.nan
            df["ML Confidence"] = np.nan
    else:
        df["ML Predicted"] = np.nan
        df["ML Confidence"] = np.nan

    return df

df = ml_prediction(df)

# =========================
# SAVE OUTPUT
# =========================
uncategorized = df[df["Categories (predicted)"].isna() & df["More Categories (less certain)"].isna() & df["ML Predicted"].isna()].copy()

# clean the columns for export
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(clean_text_for_excel)
for temp_df in [check_website, protected, uncategorized]:
    for col in temp_df.columns:
        if temp_df[col].dtype == 'object':
            temp_df[col] = temp_df[col].apply(clean_text_for_excel)

output_path = "Scraping_Results.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
    df[["Name", "Primary Key", "Website", "Categories (predicted)", "More Categories (less certain)"]].to_excel(
        writer, sheet_name="Categories Scraped", index=False)
    df.filter(items=["Name", "Primary Key", "Scraped_Text", "Website", "URLs scraped", "Scrape_Method", "Categories (predicted)", "Matched_Keywords", "More Categories (less certain)", "ML Predicted", "ML Confidence"]).to_excel(
        writer, sheet_name="Summary", index=False)
    df[df["Categories (predicted)"].isna()][["Name", "Primary Key", "Website", "Scraped_Text", "ML Predicted", "ML Confidence"]].to_excel(
        writer, sheet_name="ML Fill-In", index=False)

    uncategorized[["Name", "Primary Key", "Website", "Scraped_Text"]].to_excel(
        writer, sheet_name="Uncategorized", index=False)
    protected.to_excel(writer, sheet_name="Protected Websites", index=False)
    check_website.to_excel(writer, sheet_name="Check Website URLs", index=False)
    not_for_categorizing.to_excel(writer, sheet_name="Shouldn't Be Categorized", index=False)

    df.to_excel(writer, sheet_name="All Data", index=False)


# Download in Colab
files.download(output_path)
