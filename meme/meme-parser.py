#!/usr/bin/env python3
import re
import sys
import os
import argparse
from typing import List, Tuple, Dict

def parse_meme_file(file_path: str) -> List[Dict]:
    """
    Parse MEME output file and extract motif information.
    
    Args:
        file_path: Path to the MEME output file
        
    Returns:
        List of dictionaries containing motif information
    """
    motifs = []
    current_motif = None
    
    with open(file_path, 'r') as f:
        for line in f:
            # Look for lines starting with "MOTIF"
            if line.startswith('MOTIF'):
                # Parse the motif header line
                # Example: "MOTIF NNTGCCADN MEME-1    width =   9  sites = 1512  llr = 10197  E-value = 2.2e-1834"
                motif_match = re.match(r'MOTIF\s+(\S+)\s+(\S+)\s+width\s*=\s*(\d+)\s+sites\s*=\s*(\d+)\s+llr\s*=\s*(\d+)\s+E-value\s*=\s*(\S+)', line)
                
                if motif_match:
                    current_motif = {
                        'sequence': motif_match.group(1),
                        'id': motif_match.group(2),
                        'width': int(motif_match.group(3)),
                        'sites': int(motif_match.group(4)),
                        'llr': float(motif_match.group(5)),
                        'e_value': float(motif_match.group(6))
                    }
                    motifs.append(current_motif)
    
    return motifs

def save_to_tsv(motifs: List[Dict], output_path: str):
    """
    Save parsed motif information to a TSV file.
    
    Args:
        motifs: List of dictionaries containing motif information
        output_path: Path where the TSV file should be saved
    """
    headers = ['motif_id', 'sequence', 'width', 'sites', 'llr', 'e_value']
    
    with open(output_path, 'w') as f:
        # Write header
        f.write('\t'.join(headers) + '\n')
        
        # Write data
        for motif in motifs:
            row = [
                motif['id'],
                motif['sequence'],
                str(motif['width']),
                str(motif['sites']),
                str(motif['llr']),
                f"{motif['e_value']:.2e}"
            ]
            f.write('\t'.join(row) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Parse MEME output file and extract motif information')
    parser.add_argument('input_file', help='Path to the MEME output file')
    parser.add_argument('-o', '--output', help='Output TSV file path', default='meme_motifs.tsv')
    
    args = parser.parse_args()
    
    try:
        # Parse MEME file
        motifs = parse_meme_file(args.input_file)
        
        if not motifs:
            print("No motifs found in the input file.")
            return
        
        # Save results
        save_to_tsv(motifs, args.output)
        print(f"Successfully parsed {len(motifs)} motifs and saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
