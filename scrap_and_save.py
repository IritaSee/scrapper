from scholarly import scholarly
import pandas as pd
import time
from tqdm import tqdm
import spacy
import re
import json
from io import StringIO
from contextlib import redirect_stdout

def capture_output_contextlib(func, *args, **kwargs):
    # Create a StringIO object to capture the output
    output_buffer = StringIO()
    
    # Use the context manager to redirect stdout
    with redirect_stdout(output_buffer):
        func(*args, **kwargs)
    
    # Get the output and close the buffer
    output = output_buffer.getvalue()
    output_buffer.close()
    return output


def clean_json_string(json_str):
    # Step 1: Fix the line breaks in the abstract
    # Find the abstract portion and join it properly
    pattern = r'"abstract": "(.*?)",'  # Pattern to match the abstract content
    abstract_content = re.search(pattern, json_str, re.DOTALL)
    if abstract_content:
        # Get the abstract text
        abstract_text = abstract_content.group(1)
        # Clean up the text by removing newlines and extra spaces
        cleaned_abstract = ' '.join(
            line.strip() 
            for line in abstract_text.split('\n')
        ).strip()
        # Replace the original abstract with cleaned version
        json_str = re.sub(pattern, f'"abstract": "{cleaned_abstract}",', json_str, flags=re.DOTALL)
    
    # Step 2: Fix any remaining line breaks in the entire string
    json_str = json_str.replace('\n', '')
    
    return json_str

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
    search_query = f"{disease} treatment"
    
    try:
        # Search for papers
        print(f"Searching for papers on {search_query}...")
        search_results = scholarly.search_pubs(search_query)
        
        # Initialize progress bar
        pbar = tqdm(total=num_papers, desc="Scraping papers")
        
        # Collect data from each paper
        for i in range(num_papers):
            try:
                # Get next paper
                paper = next(search_results)
                json_output = capture_output_contextlib(scholarly.pprint, paper)
                # print(json_output[2:-2].replace("'", """\""""))
                # result = json.dumps(json_output[2:-2].replace('\n', ""))
                cleaned_json_str = clean_json_string(json_output[2:-2].replace("'", """\""""))
                print(cleaned_json_str)
                result = json.loads("'"+cleaned_json_str+"'")

                # Fill paper details
                # title = paper.get('title', 'No title available')
                # abstract = paper.get('abstract', 'No abstract available')
                # link = paper.get('url', 'No link available')
                
                title = result['bib']['title']
                abstract = result['bib']['abstract']
                # # # clean_abstract({'abstract': abstract})
                link = result['pub_url']
                print((paper.get('title')))

                # Only add papers that have abstracts
                if disease in abstract:
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
    df = scrape_research_papers(disease_name, num_papers=150)
    
    # If you want to add verb extraction (optional)
    if df is not None:
        df['Disease_Verbs'] = df['Abstract'].apply(
            lambda x: extract_disease_verbs(x, disease_name)
        )
        
        # Save updated DataFrame
        df.to_csv(f"{disease_name}_research_papers_with_verbs.csv", index=False)