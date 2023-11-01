#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:33:44 2023

@author: morgan hamm
"""
import numpy as np
import pandas as pd
import argparse
import pysam
import re
import sys

CPG_MODS = [("C", 0, "m"), ("C", 1, "m")]
M6A_MODS = [("A", 0, "a"), ("T", 1, "a"), ("A", 1, "a"), ("T", 0, "a")]


def fast_one_hot(seq_array, alphabet=list("ACGT")):
    """convert a 2d sequence array into a 3d one-hot encoded array"""
    stack_list = [np.array(seq_array == char, dtype=np.uint8) for char in alphabet]
    one_hot = np.stack(stack_list)
    one_hot = np.moveaxis(one_hot, [0, 1, 2], [1, 0, 2])
    return(one_hot)

def get_mod_bases(rec, mods=M6A_MODS):
    """
    Retrieve positions of modified bases in a record.
    
    Parameters
    ----------
    rec : pysam.libcalignedsegment.AlignedSegment
        A record containing modified bases.
    mods : list, optional
        A list of modified bases to consider, in the form (base, index, code).
        The default is M6A_MODS.

    Returns
    -------
    mod_bases : numpy.ndarray
        A sorted array of positions and scores of modified bases.
        
    Example
    -------
    mod_positions = get_mod_pos_from_rec(read.alignment)
    """
    if rec.modified_bases is None:
        return None
    mod_bases = []
    for mod in mods:
        if mod in rec.modified_bases:
            mod_score_array = np.array(
                rec.modified_bases[mod], dtype=np.int64)
            mod_bases.append(mod_score_array)
            
    if len(mod_bases) < 1:
        return None
    mod_bases = np.concatenate(mod_bases, dtype=np.int64, axis=0)
    mod_bases = mod_bases[mod_bases[:,0].argsort()]
    return mod_bases

def build_mod_seq(rec, mod_bases):
    """
    Create a sequence of modified base scores
    
    Parameters
    ----------
    rec : pysam.libcalignedsegment.AlignedSegment
        A record containing modified bases.
    mod_bases : numpy.ndarray
        An array of positions and scores of modified bases.

    Returns
    -------
    mod_seq : numpy.ndarray
        A 1D array with length equal to len(rec.query_sequence). modified base
        scores are present at their positions along the array, all other values
        are zero.
    """
    mod_seq = np.zeros(len(rec.query_sequence), dtype=int)
    for row in mod_bases:
        mod_seq[row[0]] = row[1]
    return(mod_seq)

def process_rec(rec, n_context=7, n_sites=1e8):
    """
    Parameters
    ----------
    rec : pysam.libcalignedsegment.AlignedSegment
        A record containing modified bases.
    n_context : integer
        number of bases on either side of each methylation call to include.
    n_sites : integer
        the number of sites in the record to include in the output. 
        if n_sites < # of As, sites will be picked at random. 
        default of 1e8 should ensure all As are included.
    """
    mod_bases = get_mod_bases(rec, mods=M6A_MODS)
    
    if mod_bases is None:
        return(None)
    mod_seq = build_mod_seq(rec, mod_bases) # must be done before sampling

    # use all A's instead of just modified ones
    A_locs = np.array([m.start() for m in re.finditer('A', rec.query_sequence)])
    # A_locs = np.array([i for i,c in enumerate(rec.query_sequence) if c == "A"])
    mod_bases = np.stack([A_locs, mod_seq[A_locs]], axis=1)
    
    if mod_bases.shape[0] > n_sites:
        mod_bases = mod_bases[np.random.choice(mod_bases.shape[0], n_sites, replace=False)]
        mod_bases = mod_bases[mod_bases[:,0].argsort()]
    
    # remove mods too close to start or end of sequence.
    mod_bases = mod_bases[(mod_bases[:,0] - n_context >= 0) & 
                           (mod_bases[:,0] + n_context + 1 <= len(rec.query_sequence))]
    
    if mod_bases.shape[0] == 0:
        return(None)
    
    seq_array = np.array([list(rec.query_sequence[row[0]-n_context:row[0]+n_context+1]) 
                                   for row in mod_bases])
    seq_onehot = fast_one_hot(seq_array)
    
    qual_array = np.array([list(rec.query_qualities[row[0]-n_context:row[0]+n_context+1]) 
                                   for row in mod_bases], dtype=np.uint8)/255
    
    score_array = np.array([list(mod_seq[row[0]-n_context:row[0]+n_context+1]) 
                                   for row in mod_bases], dtype=np.uint8)/255 
    
    features = np.concatenate((seq_onehot, qual_array[:, np.newaxis, :], 
                               score_array[:, np.newaxis,:]), axis=1, dtype=float)
    
    if rec.has_tag('qs'):
        qs = rec.get_tag('qs')
    else:
        qs = 0      

    return({'read_id':np.repeat(rec.query_name, features.shape[0]), 
            'features':features, 
            'length':np.repeat(len(rec.query_sequence), features.shape[0]), 
            'position':mod_bases[:, 0],
            'qs':np.repeat(qs, features.shape[0])})


def main(args):
    """
    Process a bam file containing modified bases.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace containing the following arguments:
        bam_file : str
            The path to the bam file.
        n_context : int, optional
            Number of bases on either side of each methylation call to include.
        reads_per_output: int
            number of reads to process for each output.
        sites_per_read: int
            max number of m6A sites to process for each read.
        label: int
            the label for all data from this file: 1=mixed positive, 0=negative.
        output_prefix : str
            output file name prefix.
    """
    bamfile = pysam.AlignmentFile(args.bam_file, "rb", check_sq=False, until_eof=True)
    chunk_list = []
    out_index = 0
    for i, rec in enumerate(bamfile):
        # only process unaligned data
        if rec.is_mapped:
            print(f"WARNING: {rec.query_name} is aligned, skipping.", file=sys.stderr)
            continue
            
        chunk = process_rec(rec, n_context=args.n_context, 
                                      n_sites=args.sites_per_read)
        if chunk is not None:
            chunk_list.append(chunk)
        if (i % args.reads_per_output == 0) and (i > 0):
            write_output(chunk_list, args.label, out_index, args.output_prefix)
            chunk_list = []
            out_index += 1
            print("chunk " + str(out_index - 1) + " done")
    write_output(chunk_list, args.label, out_index, args.output_prefix)
    return(None)

def write_output(chunk_list, label, out_index, output_prefix):
    """ write out to a file"""
    merged_out = {key: np.concatenate([i[key] for i in chunk_list]) for key in ['read_id', 'features', 
            'length', 'position', 'qs']}
    merged_out['label'] = np.repeat(label, merged_out['features'].shape[0])
    output_file = "{}_{:02d}.npz".format(output_prefix, out_index)
    np.savez(output_file, 
             features=merged_out['features'],
             labels=merged_out['label'],
             read_ids=merged_out['read_id'],
             lengths=merged_out['length'],
             positions=merged_out['position'],
             qs=merged_out['qs']
             )
    return(None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='parse an ONT BAM file into features surrounding called m6As')
    parser.add_argument('bam_file', help='an ONT BAM file with m6A calls')
    parser.add_argument('-n', '--n_context', type=int, default=7, 
                        help='number of bases on either side of the modified base to consider')
    parser.add_argument('-r', '--reads_per_output', type=int, default=10000, 
                        help='number of reads to process for each output')
    parser.add_argument('-s', '--sites_per_read', type=int, default=1e6, 
                        help='max number of m6A sites to process for each read')
    parser.add_argument('-l', '--label', type=int, default=1, 
                        help='the label for all data from this file: 1=mixed positive, 0=negative')   
    parser.add_argument('-o', '--output_prefix', type=str, default='output', 
                        help='output file name prefix') 
    args = parser.parse_args()
    main(args)
