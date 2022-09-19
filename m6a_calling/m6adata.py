#!/usr/bin/env python3
import argparse
import pysam
import tqdm
import numpy as np
from numba import njit
from dataclasses import dataclass, fields
import logging
import pandas as pd
import os
import pickle
from multiprocessing import Pool
from functools import partial


D_TYPE = np.int64


@dataclass
class SMRTdata:
    """A class for storing all the kinetic data associated with a PacBio subread."""

    rec: pysam.AlignedSegment
    ref_pos: np.ndarray
    read_pos: np.ndarray
    liftable_positions = np.ndarray
    ip: np.ndarray
    pw: np.ndarray
    # Conner add positions here

    def __init__(self, rec):
        self.rec = rec
        dt = np.dtype("int64,int64")
        aligned_pairs = np.array(
            rec.get_aligned_pairs(matches_only=True),
            dtype=D_TYPE,
        )
        self.read_pos = aligned_pairs[:, 0]
        self.ref_pos = aligned_pairs[:, 1]
        all_read_positions = np.arange(rec.query_length, dtype=D_TYPE)
        self.liftable_positions = np.isin(all_read_positions, self.read_pos)
        self.get_smrt_kinetics()

        assert self.ref_pos.shape == self.read_pos.shape
        assert self.liftable_positions.sum() == self.read_pos.shape[0]
        assert self.ip.shape == self.ref_pos.shape
        assert self.ip.shape == self.pw.shape

    def get_smrt_kinetics(self):
        # Conner assign new wual values here like you do for ip, pw
        ip = np.array(self.rec.get_tag("ip"), dtype=D_TYPE)
        pw = np.array(self.rec.get_tag("pw"), dtype=D_TYPE)
        if self.rec.is_reverse:
            ip = ip[::-1]
            pw = pw[::-1]
        assert ip.shape[0] == self.rec.query_length
        assert pw.shape[0] == self.rec.query_length
        self.ip = ip[self.liftable_positions]
        self.pw = pw[self.liftable_positions]

    def get_smrt_kinetics_from_positions(self, ref_pos):
        ref_pos = np.array(ref_pos, dtype=D_TYPE)
        kept = np.isin(ref_pos, self.ref_pos)
        grab = np.isin(self.ref_pos, ref_pos)
        bases = [self.rec.query_sequence[q_base] for q_base in self.read_pos[grab]]
        # return (self.ip[grab], self.pw[grab], ref_pos[kept], bases)
        return {
            "ip": self.ip[grab],
            "pw": self.pw[grab],
            "ref_pos": ref_pos[kept],
            "base": bases,
        }


@dataclass
class SMRTpileup:
    """A class for creating a pileup of all the kinetic data associated with PacBio subreads over position(s)."""

    subreads: SMRTdata
    sequence: str
    ccs_name: str
    m6a_calls: np.ndarray
    labels: int

    def __init__(
        self,
        fiber_data,
        bam,
        force_negative=False,
        min_calls=25,
    ):
        self.subreads = []
        self.ccs_name = fiber_data["fiber"]
        for rec in bam.fetch(contig=self.ccs_name):
            self.subreads.append(SMRTdata(rec))
        self.sequence = fiber_data["fiber_sequence"]
        # check for empty case

        self.m6a_calls = fiber_data["calls"]
        self.labels = fiber_data["labels"]

        if force_negative:
            self.labels = np.zeros(self.m6a_calls.shape[0], dtype=D_TYPE)

        if (
            self.m6a_calls is None
            or isinstance(self.m6a_calls, float)
            or self.m6a_calls.shape[0] < min_calls
        ):
            self.m6a_calls = None

    def get_smrt_kinetics_window(
        self, position, window_size=15, keep_all=False, keep_indels=False, label=0
    ):
        if position < 0 or position >= len(self.sequence):
            logging.info(
                f"Position {position}/{len(self.sequence)} is outside of the range of the subread sequence."
            )
            return None
        modded_base = self.sequence[position]
        extend = window_size // 2

        window = []
        for direction in -1, 1:
            count = 0
            move = 1
            while (
                True and position - move >= 0 and position + move < len(self.sequence)
            ):
                index = position + move * direction
                base = self.sequence[index]
                if base != modded_base or keep_all:
                    window.append(index)
                    count += 1
                if count == extend:
                    break
                move += 1
            # add the central base
            if direction == -1:
                window.append(position)
            # if we were unable to make a full size window skip this position
            if count != extend:
                return None
        window = sorted(window)

        rtn = []
        for smrt_data in self.subreads:
            # skip subreads from the other strand
            if modded_base == "A" and smrt_data.rec.is_reverse:
                continue
            if modded_base == "T" and not smrt_data.rec.is_reverse:
                continue
            kinetics = smrt_data.get_smrt_kinetics_from_positions(window)
            if kinetics["ip"].shape[0] == len(window) or keep_indels:
                rtn.append(kinetics)
        if len(rtn) == 0:
            return None
        df = pd.DataFrame(rtn)
        df["offset"] = df.ref_pos.apply(lambda x: x - position)

        rtn = {}
        for col in ["ip", "pw", "base", "offset"]:
            rtn[col] = np.matrix(list(df[col]))
        return SMRTmatrix(
            rtn["ip"],
            rtn["pw"],
            rtn["base"],
            rtn["offset"],
            label=label,
            ccs=self.ccs_name,
            m6a_call_base=modded_base,
            m6a_call_position=position,
        )

    def get_m6a_call_kinetics(self, window_size=15, keep_all=False):
        for label, position in zip(self.labels, self.m6a_calls):
            yield self.get_smrt_kinetics_window(
                position, window_size=window_size, keep_all=keep_all, label=label
            )


@dataclass
class SMRTmatrix:
    """A class for storing a matrix of kinetic data from PacBio subreads over a single position.
    Set up to be input for ML models.
    """

    label: int  # label for whether or not this base is an m6A call
    subread_count: int  # number of subreads that cover this position
    ccs: str  # name of the ccs read
    m6a_call_base: str  # base that is called as m6A
    m6a_call_position: int  # position of the m6A call in the ccs read
    strand: str  # strand of the m6A call
    ip: np.ndarray  # inter pulse durations (IPD)
    pw: np.matrix  # pulse width
    base: np.matrix  # sequence bases at this position
    offset: np.matrix  # offset, distance of this kinetic observation from the m6A call

    def __init__(
        self,
        ip,
        pw,
        base,
        offset,
        label=0,
        ccs="",
        m6a_call_base="",
        m6a_call_position=None,
    ):
        self.label = label
        self.subread_count = ip.shape[0]
        self.ccs = ccs
        self.m6a_call_base = "A"
        self.m6a_call_position = m6a_call_position
        self.strand = "+"

        # matrixes
        if m6a_call_base == "T":
            self.strand = "-"
            self.ip = np.fliplr(ip)
            self.pw = np.fliplr(pw)
            self.base = SMRTmatrix.revcomp_base_matrix(base)
            self.offset = np.fliplr(offset) * -1
        else:
            self.ip = ip
            self.pw = pw
            self.base = base
            self.offset = offset

    def pprint(self):
        for field in fields(self):
            print(f"{field.name}:\n{getattr(self, field.name)}\n")

    def revcomp_base_matrix(mat):
        A = mat == "A"
        T = mat == "T"
        C = mat == "C"
        G = mat == "G"
        mat[A] = "T"
        mat[T] = "A"
        mat[C] = "G"
        mat[G] = "C"
        return np.fliplr(mat)


def extend_calls(row, buffer=15, subsample=1.0):
    seq = np.frombuffer(bytes(row["fiber_sequence"], "utf-8"), dtype="S1")
    assert seq.shape[0] == len(row["fiber_sequence"])
    AT = (seq == b"A") | (seq == b"T")
    where_AT = AT.nonzero()[0]
    all_in_nuc = np.zeros(len(where_AT), dtype=bool)
    for nuc_st, nun_ln in zip(row["nuc_starts"], row["nuc_lengths"]):
        in_nuc_AT = (where_AT >= nuc_st + buffer) & (
            where_AT < nuc_st + nun_ln - buffer
        )
        all_in_nuc[in_nuc_AT] = True
    # print(all_in_nuc, all_in_nuc.sum(), all_in_nuc.shape)
    negative_label_pos = where_AT[all_in_nuc]
    # logging.info(
    #    f"{negative_labels.shape[0]} / {row['m6a'].shape[0]} negative labels/positive labels"
    # )
    calls = np.concatenate((row["m6a"], negative_label_pos), axis=None)
    labels = np.concatenate(
        (np.ones(len(row["m6a"])), np.zeros(len(negative_label_pos))), axis=None
    )
    assert len(calls) == len(labels)
    assert labels.max() < seq.shape[0]
    assert len(calls) >= len(row["m6a"])
    if subsample < 1.0:
        idxs = np.random.choice(
            np.arrange(len(calls), size=int(len(calls) * subsample), replace=False)
        )
        labels = labels[idxs]
        calls = calls[idxs]
    # print(labels)
    # print(calls)
    # print(len(calls))
    return {"labels": labels, "calls": calls}


def read_fiber_data(fiber_data_file, buffer=15, subsample=1.0):
    """Read in the fiber data file

    Args:
        fiber_data_file (str): Tabular data generated with ft extract --all

    Returns:
        pandas.DataFrame: pandas dataframe with fiber data
    """
    df = pd.read_csv(fiber_data_file, sep="\t", na_values=[".", ""])
    logging.info(f"Read {len(df)} fibers from {fiber_data_file}")
    for col in ["m6a", "nuc_starts", "nuc_lengths"]:
        df[col].fillna("", inplace=True)
        df[col] = df[col].apply(lambda x: np.fromstring(x, sep=",", dtype=D_TYPE))
    df = df.loc[(df.nuc_starts.apply(len) > 0) & (df.m6a.apply(len) > 0)]
    logging.info(f"Filtered to {len(df)} fibers")

    calls = pd.DataFrame(
        list(
            df.apply(
                lambda x: extend_calls(x, buffer=buffer, subsample=subsample), axis=1
            )
        )
    )
    logging.info(f"Generated {calls.shape[0]} calls")
    assert len(calls) == len(df)
    df = pd.concat([df.reset_index(drop=True), calls.reset_index(drop=True)], axis=1)
    df = df[df["calls"].apply(lambda x: x.shape[0]) > 1]
    logging.info(f"Filtered to {len(df)} fibers")
    return df


def read_bam(bam_file):
    """Get a handle for the bam file

    Args:
        bam_file (str): path to the bam file, needs an index

    Returns:
        pysam.AlignmentFile: handle for the bam file
    """
    assert os.path.exists(bam_file + ".bai") or os.path.exists(
        bam_file + ".csi"
    ), f"index file {bam_file}.bai does not exist"
    return pysam.AlignmentFile(bam_file, check_sq=False)


def mp_smrt_pile(
    fiber_dict, bam_file=None, force_negative=False, window_size=15, keep_all=False
):
    bam = read_bam(bam_file)
    kinetic_data = SMRTpileup(fiber_dict, bam, force_negative=force_negative)
    if kinetic_data.m6a_calls is None:
        return None
    data = []
    for t in kinetic_data.get_m6a_call_kinetics(
        keep_all=keep_all, window_size=window_size
    ):
        if t is not None:
            data.append(t)
    return data


def make_kinetic_data(bam_file, fiber_data, args):
    data = []
    mp_smrt_pile_helper = partial(
        mp_smrt_pile,
        bam_file=bam_file,
        force_negative=args.force_negative,
        window_size=args.window_size,
        keep_all=args.keep_all,
    )
    fiber_records = fiber_data.to_dict("records")
    logging.info("Processing {} fibers".format(len(fiber_records)))
    data = []
    with Pool(args.threads) as pool:
        for t in tqdm.tqdm(
            pool.imap(mp_smrt_pile_helper, fiber_records), total=len(fiber_records)
        ):
            if t is not None:
                data += t

    logging.info(f"Found {len(data)} kinetic data points.")
    out = {0: 0, 1: 0, "None": 0}
    for d in data:
        if d.label is None:
            out["None"] += 1
        else:
            out[d.label] += 1
    logging.info(f"Kinetic data labels found: {out}")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Extracts the m6A data from a BAM file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("bam", help="Input BAM file from actc")
    parser.add_argument("all", help="Input fiberseq all table")
    parser.add_argument("-o", "--out", help="Output pickle file", default=None)
    parser.add_argument("-f", "--force-negative", action="store_true")
    parser.add_argument("-k", "--keep-all", action="store_true")
    parser.add_argument("-w", "--window-size", type=int, default=15)
    parser.add_argument("-b", "--buffer", type=int, default=15)
    parser.add_argument("-t", "--threads", type=int, default=8)
    parser.add_argument("-s", "--sub-sample", type=float, default=1.0)
    args = parser.parse_args()
    log_format = "[%(levelname)s][Time elapsed (ms) %(relativeCreated)d]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    fiber_data = read_fiber_data(
        args.all, buffer=args.buffer, subsample=args.sub_sample
    )
    data = make_kinetic_data(args.bam, fiber_data, args)
    if args.out is not None:
        with open(args.out, "wb") as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    main()
