import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import os
from pathlib import Path

def parse_motifs_tsv(tsv_path):
    """
    Parse the motifs.tsv file and return sorted by sites (descending)
    """
    df = pd.read_csv(tsv_path, sep='\t')
    # Sort by sites (highest first) to get the top motifs
    df_sorted = df.sort_values('sites', ascending=False)
    return df_sorted

def extract_pwm_from_tomtom_xml(xml_path, motif_id):
    """
    Extract PWM for a specific motif_id from tomtom.xml queries section
    Returns numpy array of shape (width, 4) where columns are A, C, G, T
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Find queries section
        queries = root.find('queries')
        if queries is None:
            return None, None
        
        # Find the specific motif by alt attribute (MEME-1, MEME-2, etc.)
        target_motif = None
        for motif in queries.findall('motif'):
            if motif.get('alt') == motif_id:
                target_motif = motif
                break
        
        if target_motif is None:
            print(f"Warning: Motif {motif_id} not found in XML")
            return None, None
        
        # Extract motif metadata
        motif_info = {
            'id': target_motif.get('id'),
            'alt': target_motif.get('alt'),
            'length': int(target_motif.get('length')),
            'nsites': int(target_motif.get('nsites')),
            'evalue': target_motif.get('evalue')
        }
        
        # Extract PWM
        pwm_data = []
        for pos in target_motif.findall('pos'):
            row = [
                float(pos.get('A', 0)),
                float(pos.get('C', 0)),
                float(pos.get('G', 0)),
                float(pos.get('T', 0))
            ]
            pwm_data.append(row)
        
        pwm = np.array(pwm_data)
        return pwm, motif_info
        
    except Exception as e:
        print(f"Error extracting PWM for {motif_id}: {e}")
        return None, None

def create_meme_database(node_data_list, output_path):
    """
    Create a .meme format database file from a list of node data
    
    Parameters:
    -----------
    node_data_list : list of dicts
        Each dict should contain: 'node_id', 'pwm', 'motif_info', 'tsv_row'
    output_path : str
        Path to output .meme file
    """
    
    # MEME file header
    header = """MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

"""
    
    with open(output_path, 'w') as f:
        f.write(header)
        
        # Write each motif
        for data in node_data_list:
            node_id = data['node_id']
            pwm = data['pwm']
            motif_info = data['motif_info']
            tsv_row = data['tsv_row']
            
            # Create motif ID (use original sequence as motif name)
            motif_name = f"{node_id}_{motif_info['alt']}"
            
            # Write motif header
            f.write(f"MOTIF {motif_name}\n")
            
            # Write PWM header
            f.write(f"letter-probability matrix: alength= 4 w= {pwm.shape[0]} nsites= {motif_info['nsites']}\n")
            
            # Write PWM data
            for row in pwm:
                f.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\n")
            
            # Add URL or comment (optional)
            f.write(f"URL node_{node_id}_motif_{motif_info['alt']}\n\n")
    
    print(f"Created MEME database with {len(node_data_list)} motifs: {output_path}")

def build_meme_database_from_nodes(data_dir, output_path, max_motifs_per_node=1):
    """
    Build a MEME database from all nodes in the data directory
    
    Parameters:
    -----------
    data_dir : str
        Directory containing node subdirectories (e.g., "data_l2")
    output_path : str
        Path to output .meme database file
    max_motifs_per_node : int
        Maximum number of motifs to take per node (default 1 = top motif only)
    """
    
    all_motif_data = []
    
    # Get all node directories
    node_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    print(f"Processing {len(node_dirs)} nodes...")
    
    for node_dir in node_dirs:
        node_path = os.path.join(data_dir, node_dir)
        
        # Check for required files
        motifs_tsv_path = os.path.join(node_path, "motifs.tsv")
        tomtom_xml_path = os.path.join(node_path, "tomtom.xml")
        
        if not os.path.exists(motifs_tsv_path) or not os.path.exists(tomtom_xml_path):
            print(f"Skipping {node_dir}: missing motifs.tsv or tomtom.xml")
            continue
        
        try:
            # Parse motifs.tsv to get top motifs
            motifs_df = parse_motifs_tsv(motifs_tsv_path)
            
            # Take top N motifs
            top_motifs = motifs_df.head(max_motifs_per_node)
            
            for _, motif_row in top_motifs.iterrows():
                motif_id = motif_row['motif_id']  # e.g., "MEME-1"
                
                # Extract PWM from tomtom.xml
                pwm, motif_info = extract_pwm_from_tomtom_xml(tomtom_xml_path, motif_id)
                
                if pwm is not None and motif_info is not None:
                    all_motif_data.append({
                        'node_id': node_dir,
                        'pwm': pwm,
                        'motif_info': motif_info,
                        'tsv_row': motif_row.to_dict()
                    })
                    print(f"Added {node_dir} {motif_id} (sites: {motif_info['nsites']})")
        
        except Exception as e:
            print(f"Error processing {node_dir}: {e}")
            continue
    
    if all_motif_data:
        # Sort all motifs by number of sites (descending)
        all_motif_data.sort(key=lambda x: x['motif_info']['nsites'], reverse=True)
        
        # Create the MEME database
        create_meme_database(all_motif_data, output_path)
    else:
        print("No valid motifs found!")

def build_meme_database_single_node(node_dir, output_path, max_motifs=1):
    """
    Build a MEME database from a single node directory
    
    Parameters:
    -----------
    node_dir : str
        Path to node directory containing motifs.tsv and tomtom.xml
    output_path : str
        Path to output .meme database file
    max_motifs : int
        Maximum number of motifs to include (default 1 = top motif only)
    """
    
    motifs_tsv_path = os.path.join(node_dir, "motifs.tsv")
    tomtom_xml_path = os.path.join(node_dir, "tomtom.xml")
    
    if not os.path.exists(motifs_tsv_path) or not os.path.exists(tomtom_xml_path):
        print(f"Error: Missing motifs.tsv or tomtom.xml in {node_dir}")
        return
    
    # Parse motifs.tsv to get top motifs
    motifs_df = parse_motifs_tsv(motifs_tsv_path)
    
    # Take top N motifs
    top_motifs = motifs_df.head(max_motifs)
    
    motif_data_list = []
    
    for _, motif_row in top_motifs.iterrows():
        motif_id = motif_row['motif_id']  # e.g., "MEME-1"
        
        # Extract PWM from tomtom.xml
        pwm, motif_info = extract_pwm_from_tomtom_xml(tomtom_xml_path, motif_id)
        
        if pwm is not None and motif_info is not None:
            motif_data_list.append({
                'node_id': os.path.basename(node_dir),
                'pwm': pwm,
                'motif_info': motif_info,
                'tsv_row': motif_row.to_dict()
            })
            print(f"Added {motif_id} (sites: {motif_info['nsites']})")
    
    if motif_data_list:
        create_meme_database(motif_data_list, output_path)
    else:
        print("No valid motifs found!")

# Example usage functions
def main():
    """
    Example usage of the MEME database builder
    """
    
    # Example 1: Build database from all nodes in a directory
    print("Example 1: Building database from all nodes")
    data_dir = "data_l3"  # Your data directory
    output_path = "discovered_motifs_database_l3.meme"
    
    if os.path.exists(data_dir):
        build_meme_database_from_nodes(
            data_dir=data_dir,
            output_path=output_path,
            max_motifs_per_node=1  # Take only the top motif per node
        )
    else:
        print(f"Data directory {data_dir} not found")
    
if __name__ == "__main__":
    main()