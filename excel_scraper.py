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

# Initialize columns that will be populated by scraping to ensure they exist
for col in [
    "Website_Scraped_Text", "Scrape_Method", "Source_URL",
    "Web_categories", "Web_kw", "Web_check"
]:
    excel[col] = np.nan

# match keywords based on descriptions
excel[["Categories_Predicted", "Matched_Keywords", "Double_check"]] = excel["Scraped Text"].apply(lambda text: pd.Series(match_keywords(text, flat_keyword_map, flat_therapy_map)))

# clean and check website urls
excel["Website"] = excel["Website"].apply(clean_and_validate_url)
check_website = excel[excel["Website"].str.startswith("Check website:", na=True)][["Company Profile name", "Website"]].copy()

# find sections where categories still haven't been predicted
to_scrape = excel[
    excel["Categories_Predicted"].isna() &
    excel["Website"].notna() &
    ~excel["Website"].str.startswith("Check website:", na=False)
].copy()

if not to_scrape.empty:
    # Run first-pass website scraper
    to_scrape = await first_pass_requests(to_scrape, batch_size=25)

    # Copy scraper outputs back by index
    excel.loc[to_scrape.index, ["Website_Scraped_Text", "Scrape_Method", "Source_URL"]] = to_scrape[["Scraped_Text", "Scrape_Method", "Source_URL"]].to_numpy()

    # Clean website text after scraping
    excel.loc[to_scrape.index, "Website_Scraped_Text"] = excel.loc[to_scrape.index, "Website_Scraped_Text"].apply(clean_text)

    # Categorize website text
    excel.loc[to_scrape.index, ["Web_categories", "Web_kw", "Web_check"]] = (excel.loc[to_scrape.index, "Website_Scraped_Text"]
        .apply(lambda text: pd.Series(match_keywords(text, flat_keyword_map, flat_therapy_map)))
        .to_numpy())

# Second pass only for rows still missing BOTH description and website categories
needs_second_pass = excel[
    excel["Categories_Predicted"].isna() & (excel["Web_categories"].isna() | excel["Web_categories"].str.strip().eq("")) & 
    excel["Website"].notna() & ~excel["Website"].astype(str).str.startswith("Check website:", na=False)
].copy()
  
if not needs_second_pass.empty:
    # second_pass_uncategorized expects Scraped_Text, so create that column
    needs_second_pass["Scraped_Text"] = needs_second_pass["Website_Scraped_Text"].fillna("")

    second_pass_results = await second_pass_uncategorized(needs_second_pass)

    # Copy results back
    excel.loc[second_pass_results.index, ["Website_Scraped_Text", "Scrape_Method", "Source_URL"]] = (
        second_pass_results[["Scraped_Text", "Scrape_Method", "Source_URL"]].to_numpy())

    excel.loc[second_pass_results.index, ["Web_categories", "Web_kw", "Web_check"]] = (
        second_pass_results[["Categories_Predicted", "Matched_Keywords", "Double_check"]].to_numpy())

# combine all categories found
excel["Final_Categories"] = excel["Categories_Predicted"].combine_first(excel["Web_categories"])

# still not categorized
uncategorized = excel[excel["Categories_Predicted"].isna()].copy()

out = "Excel_scrape.xlsx"

with pd.ExcelWriter(out, engine='openpyxl', mode='w') as writer:
  excel.to_excel(writer, sheet_name="All Data", index=False)
  excel.filter(items=["Company Profile name", "Final_Categories"]).to_excel(         
      writer, sheet_name="Summary", index=False)
  excel.filter(items=["Company Profile name", "Website", "Source_URL", "Website_Scraped_Text", "Web_categories", "Web_kw", "Categories_Predicted", "Scrape_Method", "Matched_Keywords", "Double_check"]).to_excel(         
      writer, sheet_name="Specifics for debugging", index=False)
  uncategorized[["Company Profile name", "Website", "Website_Scraped_Text"]].to_excel(
               writer, sheet_name="Still Uncategorized", index=False)
  check_website.to_excel(writer, sheet_name="Check URLs", index=False)

files.download(out)
