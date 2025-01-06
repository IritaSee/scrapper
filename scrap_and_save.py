import pandas as pd
from scholarly import scholarly
from io import StringIO
from contextlib import redirect_stdout
import time
from tqdm import tqdm

def get_full_abstract(paper):
    buffer = StringIO()
    with redirect_stdout(buffer):
        scholarly.pprint(paper)
    output = buffer.getvalue()
    
    try:
        abstract_start = output.find("'abstract': '") + len("'abstract': '")
        abstract_end = output.find("',", abstract_start)
        full_abstract = output[abstract_start:abstract_end]
        full_abstract = ' '.join(line.strip() for line in full_abstract.split('\n'))
        return full_abstract
    except:
        return paper.get('bib', {}).get('abstract', '')

def scrape_disease_research(disease_name, num_papers=20):
    papers_data = []
    search_query = scholarly.search_pubs(disease_name)
    
    for _ in tqdm(range(num_papers)):
        try:
            paper = next(search_query)
            abstract = get_full_abstract(paper)
            
            if abstract:
                link = paper.get('pub_url', 'No link available')
                papers_data.append({
                    'Link': link,
                    'Abstract': abstract
                    'Title': paper.get('bib', {}).get('title', 'No title available'),
                })
            time.sleep(2)
            
        except StopIteration:
            print("\nReached end of results")
            break
        except Exception as e:
            print(f"\nError processing paper: {str(e)}")
            continue
    
    df = pd.DataFrame(papers_data)
    output_file = f"{disease_name}_research.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nData saved to {output_file}")
    return df

if __name__ == "__main__":
    disease = "glioblastoma"
    df = scrape_disease_research(disease, num_papers=20)