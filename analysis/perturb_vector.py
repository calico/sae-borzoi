import os
import json
import h5py

import pandas as pd
import numpy as np

from optparse import OptionParser
from tqdm import tqdm
from Bio import SeqIO

from baskerville import bed
from baskerville import dataset
from baskerville import dna
from baskerville import seqnn
from baskerville import snps


def hot1_shuffle(ref_1hot, shuffle_window, seed):
    """
    Shuffle the 1-hot encoded sequence within a specified window in the center.
    
    Parameters:
    ref_1hot (np.ndarray): 1-hot encoded reference sequence.
    shuffle_window (int): Size of the window to shuffle within.
    seed (int): Random seed for reproducibility.
    
    Returns:
    np.ndarray: Shuffled 1-hot encoded sequence.
    """
    np.random.seed(seed)
    seq_length = ref_1hot.shape[1]
    start = seq_length // 2 - shuffle_window // 2
    # e.g. 524288//2 - 20//2
    end = start + shuffle_window
    shuffled_seq = ref_1hot.copy()

    # test this
    shuffled_seq[:, start:end, :] = np.random.permutation(ref_1hot[:, start:end, :])

    print(f"Shuffled sequence from {start} to {end} with seed {seed}: {ref_1hot[:, start:end, :]} -> {shuffled_seq[:, start:end, :]}")

    return shuffled_seq


def create_bed_from_fasta(fasta_file, output_bed, limit_entries=None):
    """
    Convert FASTA file to BED format
    
    Parameters:
    fasta_file (str): Path to FASTA file
    output_bed (str): Path to output BED file
    limit_entries (int, optional): Limit the number of entries processed

    Returns:
    int: Number of sequences in the FASTA file
    """
    seq_count = 0
    with open(output_bed, 'w') as bed_out:
        with open(fasta_file, mode="r") as handle:
            # process each record in .fa file
            for record in SeqIO.parse(handle, "fasta"):
                identifier = record.id
                sequence = str(record.seq)
                
                # Parse the chromosome and coordinates from the identifier
                if ':' in identifier and '-' in identifier:
                    try:
                        parts = identifier.split(':')
                        chrom = parts[0]
                        start_end = parts[1].split('-')
                        start = int(start_end[0])
                        end = int(start_end[1])
                        
                        # Write to BED file (BED format is 0-based for start position)
                        bed_out.write(f"{chrom}\t{start}\t{end}\t{identifier}\t0\t+\n")
                        seq_count += 1
                    except ValueError:
                        print(f"Skipping record {identifier} in {fasta_file} due to parsing error: chr, start-end: {chrom} {start_end}")
                
                if limit_entries and seq_count >= limit_entries:
                    bed_out.close()
                    break
                
    print(f"Processed {seq_count} sequences from {fasta_file} into {output_bed}")
    

    return seq_count


def main():
    """
    perturb_vector.py

    Perform an ISM reshuffle analysis for node seqlets from .fa files.

    Usage:
        perturb_vector.py [options] <params_file> <model_file>
    Options:
        -f, --genome_fasta <file>    Genome FASTA for sequences [Default: None]
        -o, --out_dir <dir>          Output directory [Default: sat_del]
        -p, --processes <int>        Number of processes, passed by multi script
        --rc                         Ensemble forward and reverse complement predictions [Default: False]
        --num_samples <int>          Number of ISM resamples to compute [Default: 3]
        --ism_window <int>           Window size for ISM resamples [Default: 3]
        --stats <stats>              Comma-separated list of stats to save. [Default: logD2]
        -t, --targets_file <file>    File specifying target indexes and labels in table format
        --config_nodes <file>        File specifying config file for node information
        --untransform_old            Untransform old models [Default: False]
    """

    usage = "usage: %prog [options] <params_file> <model_file>"
    parser = OptionParser(usage)
    parser.add_option(
        "-f",
        dest="genome_fasta",
        default=None,
        help="Genome FASTA for sequences [Default: %default]",
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="sat_del",
        help="Output directory [Default: %default]",
    )
    parser.add_option(
        "-p",
        dest="processes",
        default=None,
        type="int",
        help="Number of processes, passed by multi script",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Ensemble forward and reverse complement predictions [Default: %default]",
    )
    parser.add_option(
        "--num_samples",
        dest="num_samples",
        default=3,
        type="int",
        help="Number of ISM resamples to compute [Default: %default]",
    )
    parser.add_option(
        "--ism_window",
        dest="ism_window",
        default=3,
        type="int",
        help="Window size for ISM resamples [Default: %default]",
    )
    parser.add_option(
        "--stats",
        dest="snp_stats",
        default="logD2",
        help="Comma-separated list of stats to save. [Default: %default]",
    )
    parser.add_option(
        "-t",
        dest="targets_file",
        default=None,
        type="str",
        help="File specifying target indexes and labels in table format",
    )
    parser.add_option(
        "--config_nodes",
        dest="config_nodes",
        default=None,
        type="str",
        help="File specifying config file for node information",
    )
    parser.add_option(
        "--untransform_old",
        dest="untransform_old",
        default=False,
        action="store_true",
        help="Untransform old models [Default: %default]",
    )
    (options, args) = parser.parse_args()

    if len(args) == 2:
        # single worker
        params_file = args[0]
        model_file = args[1]
    else:
        parser.error("Must provide parameter and model files")

    options.shifts = [int(shift) for shift in options.shifts.split(",")]
    options.snp_stats = [snp_stat for snp_stat in options.snp_stats.split(",")]

    config_file = options.config_nodes
    with open(config_file) as config_open:
        configs = json.load(config_open)

    exp_factor = configs['expansion_factor']
    topk = configs['topk_pct']
    lr = configs['learning_rate']
    global_model_path = configs['model_save_path']

    if configs['prespecified_model'] is not None:
        model_path = os.path.join(global_model_path, configs['prespecified_model'])
    else:
        model_path = os.path.join(global_model_path, f"{configs['layer_name']}_exp{exp_factor}_topk{topk}_lr{lr}")

    umap = pd.read_csv(os.path.join(model_path, 'umap_data.csv'), header=0, index_col=0)

    cwd = os.getcwd()
    temp_folder = os.path.join(cwd, "temp")

    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    node_dir = list(umap.index)
    node_dir = [f"{x}.fa" for x in node_dir]

    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]

    # create output directory if it does not exist
    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    #################################################################
    # read targets
    if options.targets_file is None:
        parser.error("Must provide targets file to clarify stranded datasets")
    targets_df = pd.read_csv(options.targets_file, sep="\t", index_col=0)

    if options.target_genes is not None:
        target_genes = pd.read_csv(
            options.target_genes, sep="\t", index_col=None, header=None
        )
        target_genes.columns = ["gene_id"]

    # handle strand pairs
    if "strand_pair" in targets_df.columns:
        # prep strand
        targets_strand_df = dataset.targets_prep_strand(targets_df)

        # set strand pairs (using new indexing)
        orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
        targets_strand_pair = np.array(
            [orig_new_index[ti] for ti in targets_df.strand_pair]
        )
        params_model["strand_pair"] = [targets_strand_pair]

        # construct strand sum transform
        strand_transform = dataset.make_strand_transform(targets_df, targets_strand_df)
    else:
        targets_strand_df = targets_df
        strand_transform = None
    num_targets = targets_strand_df.shape[0]


    #################################################################
    # setup model

    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file)
    seqnn_model.build_slice(targets_df.index)
    seqnn_model.build_ensemble(options.rc)

    # helper variables
    seq_mid = params_model["seq_length"] // 2
    model_stride = seqnn_model.model_strides[0]

    for node_id in tqdm(node_dir, desc="Creating BED files"):

        #################################################################
        # make bed file from fasta

        node_fasta = f"{model_path}/node_seqs_test_1000/{node_id}"
        node_name = node_id.split(".")[0]
        node_bed = os.path.join(temp_folder, f"{node_name}.bed")
        
        # Create BED file and get sequence count
        _ = create_bed_from_fasta(node_fasta, node_bed)

        # read sequences from BED
        seqs_dna, seqs_coords, ism_lengths = bed.make_bed_seqs(
            node_bed, options.genome_fasta, params_model["seq_length"], stranded=True
        )

        #################################################################
        # setup scores h5 file

        scores_h5_file = f"{options.out_dir}/scores_{node_id}.h5"
        if os.path.isfile(scores_h5_file):
            os.remove(scores_h5_file)
        scores_h5 = h5py.File(scores_h5_file, "w")
        scores_h5.create_dataset(
            "seqs", dtype="bool", shape=(len(seqs_dna), ism_lengths, 4)
        )
        print("Seqs shape:", (len(seqs_dna), ism_lengths, 4))
        for snp_stat in options.snp_stats:
            scores_h5.create_dataset(
                snp_stat, dtype="float16", shape=(len(seqs_dna), num_targets)
            )

        #################################################################
        # make preds

        for si, seq_dna in enumerate(seqs_dna):
            ref_1hot = dna.dna_1hot(seq_dna)
            ref_1hot = np.expand_dims(ref_1hot, axis=0)

            ref_preds = seqnn_model.predict_transform(
                ref_1hot,
                targets_df,
                strand_transform,
                options.untransform_old,
            )

            # predict reference
            alt_preds = []
            for sample in range(options.num_samples):
                # reshuffle sequence, 1hot and predict
                shuffle_seed = np.random.randint(0, 2**32 - 1)
                alt_1hot = hot1_shuffle(ref_1hot, options.ism_window, shuffle_seed)
                
                alt_pred = seqnn_model.predict_transform(
                    alt_1hot,
                    targets_df,
                    strand_transform,
                    options.untransform_old,
                )
                alt_preds.append(alt_pred)

            alt_preds = np.array(alt_preds)
            alt_preds = np.mean(alt_preds, axis=0)

            ism_scores = snps.compute_scores(
                ref_preds, alt_preds, options.snp_stats, None
            )
            for snp_stat in options.snp_stats:
                scores_h5[snp_stat][si, :] = ism_scores[snp_stat]

        scores_h5.close()