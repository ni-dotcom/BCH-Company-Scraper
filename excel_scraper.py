# scrape additional companies with descriptions in excel
# use with main_scraper.py

excel = pd.read_excel("CompaniesExport.xlsx")

excel["Scraped Text"] = (
    excel["Company type(s)"].fillna("").astype(str) + "\n" +
    excel["Brief description"].fillna("").astype(str) + "\n" +
    excel["Description"].fillna("").astype(str) + "\n" +
    excel["Primary therapeutic area(s)"].fillna("").astype(str) + "\n" +
    excel["Secondary therapeutic area(s)"].fillna("").astype(str) + "\n" +
    excel["Partnering objectives"].fillna("").astype(str) + "\n" +
    excel["Company objectives"].fillna("").astype(str)
)

excel[["Categories_Predicted", "Matched_Keywords", "Double_check"]] = excel["Scraped Text"].apply(lambda text: pd.Series(match_keywords(text, flat_keyword_map, flat_therapy_map)))

# find sections where categories still haven't been predicted
unlabeled = excel[excel["Categories_Predicted"].isna()]
if not unlabeled.empty:
  # scrape websites for these companies
  website_text = unlabeled["Website"].apply(clean_and_validate_url).apply(scrape_relevant_sections)
  excel.loc[unlabeled.index, "Website_Scraped_Text"] = website_text

  # clean text
  for col in excel.columns:
    if excel[col].dtype == 'object': # object data type indicates it contains strings or mixed types
        excel[col] = excel[col].apply(clean_text)

  excel[["Web_categories", "Web_kw", "Web_check"]] = excel["Website_Scraped_Text"].apply(lambda text: pd.Series(match_keywords(text, flat_keyword_map, flat_therapy_map)))

else:
  excel[["Web_categories", "Web_kw", "Web_check"]] = np.nan, np.nan, np.nan

out = "Excel_scrape.xlsx"
with pd.ExcelWriter(out, engine='openpyxl', mode='w') as writer:
  excel.to_excel(writer, sheet_name="All Data", index=False)
  excel[["Scraped Text", "Company Profile name", "Website", "Categories_Predicted", "Matched_Keywords", "Double_check", "Web_categories", "Web_check", "Website_Scraped_Text"]].to_excel(writer, sheet_name="Specific", index=False)

files.download(out)
