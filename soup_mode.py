import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import time
import urllib.parse

def scrape_research_papers(disease, num_pages=2):
    base_url = "https://pubmed.ncbi.nlm.nih.gov/"
    search_url = f"https://pubmed.ncbi.nlm.nih.gov/?term={urllib.parse.quote(disease)}"
    papers_data = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for page in tqdm(range(1, num_pages + 1)):
        try:
            page_url = f"{search_url}&page={page}"
            response = requests.get(page_url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.find_all('article', class_='full-docsum')

            for article in articles:
                try:
                    title_element = article.find('a', class_='docsum-title')
                    title = title_element.text.strip() if title_element else "No title available"
                    paper_id = title_element['href'].strip('/') if title_element else None
                    
                    if paper_id:
                        paper_url = base_url + paper_id
                        paper_response = requests.get(paper_url, headers=headers)
                        paper_soup = BeautifulSoup(paper_response.text, 'html.parser')
                        
                        abstract_element = paper_soup.find('div', class_='abstract-content')
                        abstract = abstract_element.text.strip() if abstract_element else "No abstract available"
                        
                        papers_data.append({
                            'Title': title,
                            'Abstract': abstract,
                            'Link': paper_url
                        })
                        
                        time.sleep(1)  # Respect rate limits
                
                except Exception as e:
                    print(f"Error processing article: {str(e)}")
                    continue
            
            time.sleep(2)  # Delay between pages
            
        except Exception as e:
            print(f"Error processing page {page}: {str(e)}")
            continue

    df = pd.DataFrame(papers_data)
    output_file = f"{disease}_research_pubmed.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nData saved to {output_file}")
    return df

if __name__ == "__main__":
    disease = "glioblastoma treatment"
    df = scrape_research_papers(disease, num_pages=10)