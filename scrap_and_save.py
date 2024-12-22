from scholarly import scholarly
import pandas as pd
import time
from tqdm import tqdm
import spacy
import re

def scrape_research_papers(disease, num_papers=50):
    """
    Scrapes research papers from Google Scholar related to a specific disease.
    
    Parameters:
    disease (str): The disease to search for
    num_papers (int): Number of papers to retrieve (default: 50)
    
    Returns:
    pandas.DataFrame: DataFrame containing paper information
    """
    # Initialize empty lists to store our data
    titles = []
    abstracts = []
    links = []
    
    # Create the search query
    search_query = f"{disease} treatment research"
    
    try:
        # Search for papers
        search_results = scholarly.search_pubs(search_query)
        
        # Initialize progress bar
        pbar = tqdm(total=num_papers, desc="Scraping papers")
        
        # Collect data from each paper
        for i in range(num_papers):
            try:
                # Get next paper
                paper = next(search_results)
                
                # Fill paper details
                title = paper.get('title', 'No title available')
                abstract = paper.get('abstract', 'No abstract available')
                link = paper.get('url', 'No link available')
                
                # Only add papers that have abstracts
                if abstract != 'No abstract available':
                    titles.append(title)
                    abstracts.append(abstract)
                    links.append(link)
                    
                    # Update progress bar
                    pbar.update(1)
                
                # Add delay to avoid overwhelming the server
                time.sleep(2)
                
            except StopIteration:
                print(f"\nReached end of results after {i} papers")
                break
            except Exception as e:
                print(f"\nError processing paper: {str(e)}")
                continue
                
        pbar.close()
        
    except Exception as e:
        print(f"Error during search: {str(e)}")
        return None
    
    # Create DataFrame
    df = pd.DataFrame({
        'Title': titles,
        'Abstract': abstracts,
        'Link': links
    })
    
    # Save to CSV
    output_filename = f"{disease}_research_papers.csv"
    df.to_csv(output_filename, index=False)
    print(f"\nData saved to {output_filename}")
    
    return df

def extract_disease_verbs(text, disease):
    """
    Extracts verbs associated with the disease in the text.
    This is a basic implementation that can be enhanced later.
    
    Parameters:
    text (str): The text to analyze (abstract)
    disease (str): The disease name to look for
    
    Returns:
    list: List of verbs associated with the disease
    """
    # Load English language model
    nlp = spacy.load("en_core_web_sm")
    
    # Process the text
    doc = nlp(text)
    
    # Find sentences containing the disease name
    disease_sentences = [sent.text for sent in doc.sents if disease.lower() in sent.text.lower()]
    
    # Extract verbs from these sentences
    verbs = []
    for sentence in disease_sentences:
        sent_doc = nlp(sentence)
        # Get verbs that appear near the disease name
        verbs.extend([token.text for token in sent_doc if token.pos_ == "VERB"])
    
    return list(set(verbs))  # Remove duplicates

# Example usage
if __name__ == "__main__":
    # Set the disease you want to search for
    disease_name = "glioblastoma"
    
    # Scrape the papers
    df = scrape_research_papers(disease_name, num_papers=10)
    
    # If you want to add verb extraction (optional)
    if df is not None:
        df['Disease_Verbs'] = df['Abstract'].apply(
            lambda x: extract_disease_verbs(x, disease_name)
        )
        
        # Save updated DataFrame
        df.to_csv(f"{disease_name}_research_papers_with_verbs.csv", index=False)