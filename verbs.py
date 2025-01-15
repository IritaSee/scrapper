import pandas as pd
import spacy
import re
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

def visualize_disease_verbs(df, disease_name, filename):
    """
    Create a network visualization of verbs connected to the disease.
    
    Args:
        df: DataFrame containing the verb analysis results
        disease_name: Name of the disease being analyzed
    """
    # Create a new graph
    G = nx.Graph()
    
    # Add disease node as central node
    G.add_node(disease_name, node_type='disease')
    
    # Get verb frequencies
    verb_counts = df['Verb'].value_counts()
    max_count = verb_counts.max()
    
    # Add verb nodes and edges
    edge_weights = {}
    for verb, count in verb_counts.items():
        # Calculate node size based on frequency
        size = (count / max_count) * 1000 + 100
        # print(f"{verb}: {count}")
        G.add_node(verb, node_type='verb', size=size)
        G.add_edge(disease_name, verb, weight=count)
        edge_weights[(disease_name, verb)] = count
    
    # Set up the plot
    plt.figure(figsize=(15, 15))
    
    # Create the layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes
    node_colors = ['red' if data['node_type'] == 'disease' else 'lightblue' 
                  for _, data in G.nodes(data=True)]
    node_sizes = [1500 if data['node_type'] == 'disease' else data.get('size', 300) 
                 for _, data in G.nodes(data=True)]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    nx.draw_networkx_labels(G, pos, font_size=6)

    # Add edge labels (occurrence counts)
    # edge_labels = {(u, v): f'{d["weight"]:.0f}' 
                #   for (u, v, d) in G.edges(data=True)}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)

    plt.title(f"Verbs associated with {disease_name}")
    plt.axis('off')
    
    # Save the plot
    output_file = f"{filename}_network.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\nNetwork visualization saved as {output_file}")


def extract_verbs_after_disease(text, disease, nlp):
    """Extract verbs that appear after the disease mention within a window."""
    # Remove the quotes from the text if they exist
    text = text.strip('"')
    doc = nlp(text.lower())
    verbs = []
    
    for sent in doc.sents:
        sent_doc = nlp(sent.text)
        # Find all occurrences of the disease in this sentence
        disease_indices = [i for i, token in enumerate(sent_doc) 
                         if disease.lower() in token.text.lower()]
        
        for idx in disease_indices:
            # Look at the next 5 tokens after the disease mention
            end_idx = min(idx + 10, len(sent_doc))
            following_tokens = sent_doc[idx+1:end_idx]
            
            # Find the first verb in the window
            for token in following_tokens:
                if token.pos_ == "VERB":
                    # Store both the verb and its context
                    context = sent_doc[max(0, idx-2):min(len(sent_doc), idx+10)].text
                    verbs.append({
                        'verb': token.lemma_,
                        'original_form': token.text,
                        'context': context
                    })
                    break
    
    return verbs

def process_csv_for_verbs(csv_file, disease_name):
    # Load spaCy model
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    
    # Read the CSV file
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Initialize lists to store results
    all_results = []
    
    # Process each row
    print("Processing abstracts...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        title_verbs = extract_verbs_after_disease(row['Title'], disease_name, nlp)
        abstract_verbs = extract_verbs_after_disease(row['Abstract'], disease_name, nlp)
        
        # Combine results from title and abstract
        if title_verbs or abstract_verbs:
            all_results.append({
                'Title': row['Title'],
                'Link': row['Link'],
                'Title_Verbs': title_verbs,
                'Abstract_Verbs': abstract_verbs
            })
    
    # Create output DataFrame with verb information
    output_data = []
    for result in all_results:
        # Process title verbs
        for verb_info in result['Title_Verbs']:
            output_data.append({
                'Source': 'Title',
                'Text': result['Title'],
                'Link': result['Link'],
                'Verb': verb_info['verb'],
                'Original_Form': verb_info['original_form'],
                'Context': verb_info['context']
            })
        
        # Process abstract verbs
        for verb_info in result['Abstract_Verbs']:
            output_data.append({
                'Source': 'Abstract',
                'Text': result['Title'],  # Using title as reference
                'Link': result['Link'],
                'Verb': verb_info['verb'],
                'Original_Form': verb_info['original_form'],
                'Context': verb_info['context']
            })
    
    # Create and save the results
    output_df = pd.DataFrame(output_data)
    output_file = f"{csv_file}_following_verbs.csv"
    output_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\nSummary of verbs found:")
    print(output_df)
    verb_counts = output_df['Verb'].value_counts()
    
    # verb_counts = output_df
    verb_counts.to_csv(f"{csv_file}_verb_counts.csv", header=True)
    print(verb_counts.head(10))

    visualize_disease_verbs(output_df, disease_name, csv_file)
    
    return output_df

if __name__ == "__main__":
    csv_file = "glioblastoma invasiveness_research_pubmed_quoted_100.csv"  # Your input CSV file
    disease = "glioblastoma"
    results_df = process_csv_for_verbs(csv_file, disease)