# scrape additional companies with descriptions in excel
# use with main_scraper.py

excel = pd.read_excel("CompaniesExport.xlsx")

excel["Scraped Text"] = excel["Company type(s)"].astype(str) + "\n" + excel["Brief description"].astype(str) + "\n" + excel["Description"].astype(str) + "\n" + excel["Primary therapeutic area(s)"].astype(str) + "\n" + excel["Secondary therapeutic area(s)"].astype(str) + "\n" + excel["Partnering objectives"].astype(str) + "\n" + excel["Company objectives"].astype(str)

excel[["Categories_Predicted", "Matched_Keywords", "Double_check"]] = excel["Scraped Text"].apply(lambda text: pd.Series(match_keywords(text, flat_keyword_map, flat_therapy_map)))

out = "Excel_scrape.xlsx"
with pd.ExcelWriter(out, engine='openpyxl', mode='w') as writer:
  excel.to_excel(writer, sheet_name="All Data", index=False)
  excel[["Scraped Text", "Company Profile name", "Website", "Categories_Predicted", "Matched_Keywords", "Double_check"]].to_excel(writer, sheet_name="Specific", index=False)

files.download(out)
