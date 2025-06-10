import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from pathlib import Path

def load_existing_dois(output_path):
    if output_path.exists():
        with open(output_path, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def scrape_pubmed_dois(query, max_results=100, year_range=(2015, 2025), existing_dois=None):
    if existing_dois is None:
        existing_dois = set()

    new_dois = []
    base_url = "https://pubmed.ncbi.nlm.nih.gov"
    headers = {"User-Agent": "Mozilla/5.0"}

    page = 1
    collected = 0

    while collected < max_results:
        search_url = (
            f"{base_url}/?term={quote_plus(query)}"
            f"+AND+({year_range[0]}%3A{year_range[1]}[dp])"
            f"&page={page}"
        )
        response = requests.get(search_url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to retrieve PubMed results for page {page}.")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        results = soup.find_all("article", class_="full-docsum")
        if not results:
            break

        for result in results:
            if collected >= max_results:
                break

            try:
                title_tag = result.find("a", class_="docsum-title")
                if not title_tag:
                    continue

                article_url = base_url + title_tag['href']
                article_page = requests.get(article_url, headers=headers)
                article_soup = BeautifulSoup(article_page.text, 'html.parser')

                doi_tag = article_soup.find("span", class_="citation-doi")
                if doi_tag:
                    doi = doi_tag.get_text(strip=True).replace("doi: ", "").rstrip(".;, ")
                    if doi not in existing_dois and doi not in new_dois:
                        new_dois.append(doi)
                        collected += 1
                        print(f"✅ New DOI found: {doi}")
                    else:
                        print(f"⏩ Skipping duplicate DOI: {doi}")
            except Exception as e:
                print(f"Error scraping DOI: {e}")
                continue

        page += 1

    return new_dois

def save_dois_to_txt(dois, output_path):
    if not dois:
        print("No new DOIs to save.")
        return

    with open(output_path, "a") as f:
        for doi in dois:
            f.write(f"{doi}\n")
    print(f"\n✅ Appended {len(dois)} new DOIs to: {output_path}")

if __name__ == "__main__":
    keywords = [
        "ctg", "cardiotocography", "fetal monitoring", "nonstress test", "fetal heart rate",
        "tocogram", "prenatal care", "perinatal outcome", "obstetrics", "neonatal outcome",
        "preterm birth", "apgar score", "fetal growth restriction", "ultrasound", "biophysical profile",
        "anomaly scan", "risk prediction", "machine learning", "deep learning", "ai", "predictive model",
        "maternal-fetal medicine", "nicu", "postnatal care", "birth complications", 'ctg classification',
        'ctg interpretation', 'ctg prediction', 'fetal distress', 'uterine contraction', 'maternal-fetal interface',
        'placental insufficiency', 'doppler velocimetry', 'fetal echocardiography', 'cardiac anomalies',
        'genetic screening', 'maternal health', 'fetal arrhythmia', 'oxytocin challenge test', 'labor monitoring',
        'birth risk scoring', 'fetal oxygenation', 'biophysical scoring', 'intrapartum monitoring', 'intrapartum care',
        'maternal morbidity', 'neonatal complications', 'ctg feature extraction', 'ai-assisted obstetrics'
    ]
    query = " OR ".join(keywords)

    project_root = Path(__file__).resolve().parents[1]
    save_dois_path = project_root / "Academic Paper Storage" / "ctg_dois.txt"

    existing_ctg_dois = load_existing_dois(save_dois_path)
    new_ctg_dois = scrape_pubmed_dois(query, max_results=1000, year_range=(2015, 2025), existing_dois=existing_ctg_dois)
    save_dois_to_txt(new_ctg_dois, save_dois_path)
