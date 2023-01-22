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
import gnuplotlib as gp


# Global statics
CPG_MODS = [("C", 0, "m")]
M6A_MODS = [("A", 0, "a"), ("T", 1, "a")]
D_TYPE = np.int64


@njit
def convert_u16_to_u8(x):
    z = np.zeros(x.shape[0], dtype=np.uint8)
    for i, v in enumerate(x):
        if v < 64:
            z[i] = v
        elif v < 191:
            z[i] = (v - 64) // 2 + 64
        elif v < 445:
            z[i] = (v - 192) // 4 + 128
        elif v < 953:
            z[i] = (v - 448) // 8 + 192
        else:
            z[i] = 255
    return z


@dataclass
class SMRTdata:
    """A class for storing all the kinetic data associated with a PacBio subread."""

    rec: pysam.AlignedSegment
    ref_pos: np.ndarray
    read_pos: np.ndarray
    liftable_positions = np.ndarray
    ip: np.ndarray
    pw: np.ndarray
    is_u16: bool
    # Conner add positions here

    def __init__(self, rec, is_u16=False):
        self.is_u16 = is_u16
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
        if self.is_u16:
            ip = convert_u16_to_u8(ip)
            pw = convert_u16_to_u8(pw)

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
        is_u16=False,
    ):
        self.subreads = []
        self.ccs_name = fiber_data["fiber"]
        for rec in bam.fetch(contig=self.ccs_name):
            self.subreads.append(SMRTdata(rec, is_u16=is_u16))
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
            if modded_base == "T" and smrt_data.rec.is_reverse:
                continue
            if modded_base == "A" and not smrt_data.rec.is_reverse:
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
            np.arange(len(calls)), size=int(len(calls) * subsample), replace=False
        )
        labels = labels[idxs]
        calls = calls[idxs]
    return {"labels": labels, "calls": calls}


def read_fiber_data(fiber_data_file, bam_file, buffer=15, subsample=1.0):
    """Read in the fiber data file

    Args:
        fiber_data_file (str): Tabular data generated with ft extract --all

    Returns:
        pandas.DataFrame: pandas dataframe with fiber data
    """
    bam = read_bam(bam_file)
    ccs_reads_to_keep = bam.references

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
    df = df[df["fiber"].isin(ccs_reads_to_keep)]
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
    fiber_dict,
    bam_file=None,
    force_negative=False,
    window_size=15,
    keep_all=False,
    is_u16=False,
):
    bam = read_bam(bam_file)
    kinetic_data = SMRTpileup(
        fiber_dict, bam, force_negative=force_negative, is_u16=is_u16
    )
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
        is_u16=args.is_u16,
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


@dataclass
class SMRThifi:
    """A class for storing all the kinetic data associated with a PacBio hifi."""

    rec: pysam.AlignedSegment
    f_ip: np.ndarray
    f_pw: np.ndarray
    r_ip: np.ndarray
    r_pw: np.ndarray
    f_m6a: np.ndarray
    r_m6a: np.ndarray
    m6a_calls: np.ndarray
    nuc_starts: np.ndarray
    nuc_lengths: np.ndarray
    labels: np.ndarray
    positions: np.ndarray

    def __init__(
        self,
        rec,
        min_nuc_bp=2000,
        min_nucs=10,
        train=False,
        buffer=15,
        is_u16=False,
        min_ml_score=200,
    ):
        self.rec = rec
        self.is_u16 = is_u16
        self.seq = np.frombuffer(bytes(self.rec.query_sequence, "utf-8"), dtype="S1")
        self.f_ip = self.get_tag("fi")
        self.f_pw = self.get_tag("fp")
        self.r_ip = self.get_tag("ri")[::-1]
        self.r_pw = self.get_tag("rp")[::-1]
        if self.is_u16:
            self.f_ip = convert_u16_to_u8(self.f_ip)
            self.f_pw = convert_u16_to_u8(self.f_pw)
            self.r_ip = convert_u16_to_u8(self.r_ip)
            self.r_pw = convert_u16_to_u8(self.r_pw)

        self.get_mod_pos_from_rec(min_ml_score=min_ml_score)
        # logging.info(f"{self.nuc_starts.shape}")

        labels = np.zeros(len(self.seq), dtype=bool)
        if len(self.f_m6a) > 0:
            labels[self.f_m6a] = True
        if len(self.r_m6a) > 0:
            labels[self.r_m6a] = True
        self.labels = labels

        # get average kinetics for the reads
        pw_means = []
        ip_means = []
        for base in [b"A", b"C", b"G", b"T"]:
            is_base = self.seq == base
            if is_base.sum() > 0 and self.f_pw.shape[0] > 0 and self.f_ip.shape[0] > 0:
                pw_means.append(np.median(self.f_pw[is_base]) / 255.0)
                ip_means.append(np.median(self.f_ip[is_base]) / 255.0)
            else:
                pw_means.append(0)
                ip_means.append(0)
        self.pw_means = np.array(pw_means)
        self.ip_means = np.array(ip_means)

        # set up positions we will call
        positions = np.arange(len(self.seq))
        if train:
            self.nuc_starts = self.get_tag("ns")
            self.nuc_lengths = self.get_tag("nl")
            if (
                self.nuc_lengths.sum() < min_nuc_bp
                or self.nuc_lengths.shape[0] < min_nucs
            ):
                logging.debug("Too few nucleosomes or nucleosome bases")
                self.m6a_calls = None
                self.f_ip = np.array([])
                self.r_ip = np.array([])
                self.f_pw = np.array([])
                self.r_pw = np.array([])
                keep = 1
            else:
                logging.debug(f"{self.nuc_starts.shape}")
                keep = SMRThifi.filter_negatives_by_nucleosomes(
                    positions, self.nuc_starts, self.nuc_lengths, self.labels, buffer
                )
                logging.debug(
                    f"{keep.sum()} positions kept, {len(keep) - keep.sum()} positions removed"
                )
        else:
            keep = 1
        is_AT = (self.seq == b"A") | (self.seq == b"T")
        self.positions = positions[keep & is_AT]

        self.m6a_calls = None
        # self.m6a_calls = self.get_mod_pos_from_rec()

    @njit
    def filter_negatives_by_nucleosomes(
        positions, nuc_start, nuc_lengths, labels, buffer
    ):
        # keep positives
        keep = labels.copy()
        # add negatives in nucleosome regions
        for st, ln in zip(nuc_start, nuc_lengths):
            keep[(positions >= st + buffer) & (positions < st + ln - buffer)] = True
        return keep

    def get_tag(self, tag):
        if self.rec.has_tag(tag):
            return np.array(self.rec.get_tag(tag), dtype=np.int64)
        logging.info(f"Missing {tag}")
        return np.array([], dtype=np.int64)

    def get_mod_pos_from_rec(self, mods=M6A_MODS, min_ml_score=200):
        self.f_m6a = np.array([])
        self.r_m6a = np.array([])
        positions = []
        for mod in mods:
            if mod in self.rec.modified_bases_forward:
                pos = np.array(self.rec.modified_bases_forward[mod], dtype=D_TYPE)[:, 0]
                mls = np.array(self.rec.modified_bases_forward[mod], dtype=D_TYPE)[:, 1]
                if mod[1] == 0:
                    self.f_m6a = pos[mls >= min_ml_score]
                    # print(self.f_m6a)
                elif mod[1] == 1:
                    self.r_m6a = pos[mls >= min_ml_score]
                    # print("here2")
                positions.append(pos)
        if len(positions) < 1:
            return None
        mod_positions = np.concatenate(positions, dtype=D_TYPE)
        mod_positions.sort(kind="mergesort")
        return mod_positions

    def get_windows(self, window_size=15, subsample=1, buffer=30):
        return SMRThifi.get_windows_helper(
            self.seq,
            self.rec.query_name,
            self.labels,
            self.positions,
            self.f_m6a,
            self.r_m6a,
            self.f_ip,
            self.f_pw,
            self.r_ip,
            self.r_pw,
            window_size,
            subsample,
            buffer,
        )

    @njit()
    def get_windows_helper(
        self_seq,
        fiber,
        labels,
        positions,
        self_f_m6a,
        self_r_m6a,
        self_f_ip,
        self_f_pw,
        self_r_ip,
        self_r_pw,
        window_size,
        subsample,
        buffer,
    ):
        if labels.sum() == 0:
            return None

        extend = window_size // 2
        Ab = np.frombuffer(b"A", dtype="S1")
        Cb = np.frombuffer(b"C", dtype="S1")
        Gb = np.frombuffer(b"G", dtype="S1")
        Tb = np.frombuffer(b"T", dtype="S1")

        if subsample < 1:
            positions = np.random.choice(
                positions, size=int(len(positions) * subsample), replace=False
            )

        out_labels = []
        strands = []
        windows = []
        out_positions = []
        pre_label = -1
        pre_pos = -1
        for pos, base in zip(positions, self_seq[positions]):
            base = self_seq[pos]
            label = labels[pos]
            # skip is we are too close to a positive label
            # if label == 0 and pre_label == 1 and pos - pre_pos < buffer:
            #    continue
            if not (base == b"A" or base == b"T"):
                continue
            start = pos - extend
            end = pos + extend + 1
            if start < 0 or end > len(self_seq):
                continue
            seq = self_seq[start:end]

            At = seq == Ab
            Ct = seq == Cb
            Gt = seq == Gb
            Tt = seq == Tb
            # if the incorporated base is A then revcomp the sequence and values
            if base == b"A":
                strand = 1
                # need to reverse again after grabbing profiles.
                # not 100% sure why however, empirically it makes the PW and IPD align better.
                ip = self_r_ip[start:end][::-1] / 255.0
                pw = self_r_pw[start:end][::-1] / 255.0
                T = At[::-1]
                G = Ct[::-1]
                C = Gt[::-1]
                A = Tt[::-1]
            elif base == b"T":
                strand = 0
                ip = self_f_ip[start:end] / 255.0
                pw = self_f_pw[start:end] / 255.0
                A = At
                C = Ct
                G = Gt
                T = Tt

            if ip.shape[0] != window_size or pw.shape[0] != window_size:
                continue
            window = np.vstack((A, C, G, T, ip, pw))
            windows.append(window)
            strands.append(strand)
            out_labels.append(label)
            out_positions.append(pos)
            pre_label = label
            pre_pos = pos
        return out_labels, strands, windows, out_positions, [fiber] * len(windows)


def make_hifi_kinetic_data_helper(rec, args=None):
    hifi = SMRThifi(
        rec,
        buffer=args.buffer,
        train=args.train,
        min_nuc_bp=args.min_nuc_bp,
        min_nucs=args.min_nucs,
        is_u16=args.is_u16,
        min_ml_score=args.min_ml_score,
    )
    logging.debug(f"{hifi}")
    if hifi is None or hifi.f_ip.shape[0] == 0 or hifi.r_ip.shape[0] == 0:
        return None
    data = hifi.get_windows(
        window_size=args.window_size, subsample=args.sub_sample, buffer=args.buffer
    )
    if data is None:
        return None
    return data + (len(data[0]) * [hifi.pw_means], len(data[0]) * [hifi.ip_means])


def make_hifi_kinetic_data(bam_file, args):
    logging.info(f"Reading HiFi {bam_file}")
    bam = pysam.AlignmentFile(bam_file, check_sq=False, threads=args.threads)
    labels = []
    strands = []
    windows = []
    positions = []
    fibers = []
    pw_means = []
    ip_means = []
    for idx, rec in tqdm.tqdm(enumerate(bam.fetch(until_eof=True))):
        if rec.get_tag("ec") < args.ec:
            continue
        data = make_hifi_kinetic_data_helper(rec, args)
        if data is not None:
            labels += data[0]
            strands += data[1]
            windows += data[2]
            positions += data[3]
            fibers += data[4]
            pw_means += data[5]
            ip_means += data[6]

    for z in [labels, strands, windows, positions, fibers]:
        logging.info(f"{len(z)}")

    labels = np.array(labels)
    strands = np.array(strands)
    windows = np.array(windows)
    positions = np.array(positions)
    fibers = np.array(fibers)
    ip_means = np.array(ip_means)
    pw_means = np.array(pw_means)
    for z in [labels, strands, windows, positions, fibers, pw_means, ip_means]:
        logging.info(f"{z.shape}")

    for strand in [0, 1]:
        central_ip = (
            255 * windows[labels & (strand == strands), 4, args.window_size // 2]
        )
        non_m6a = 255 * windows[~labels & (strand == strands), 4, args.window_size // 2]
        p_base_ave = 255 * windows[labels & (strand == strands), 0:4, 5:10].mean(axis=0)
        n_base_ave = 255 * windows[~labels & (strand == strands), 0:4, 5:10].mean(
            axis=0
        )
        logging.info(f"Strand: {strand}")

        logging.info(
            f"Mean IPD at m6A:\t{central_ip.mean():.4g} +/- {central_ip.std():.4g}"
        )
        logging.info(f"Base weights:\n{p_base_ave}")
        logging.info(f"Base weights:\n{n_base_ave}")
        logging.info(
            f"Mean IPD at non-m6A:\t{non_m6a.mean():.4g} +/- {non_m6a.std():.4g}"
        )
        logging.info("")
        pos = np.mean(255 * windows[labels & (strand == strands), 4, :], axis=0)
        neg = np.mean(255 * windows[~labels & (strand == strands), 4, :], axis=0)
        gp.plot(
            pos,
            _with="lines",
            terminal="dumb 80,30",
            unset="grid",
            title=f"IPD at m6A on the {strand}",
        )
        gp.plot(
            neg,
            _with="lines",
            terminal="dumb 80,30",
            unset="grid",
            title=f"IPD at non-m6A on the {strand}",
        )

    np.savez_compressed(
        args.out,
        features=windows,
        labels=labels,
        strands=strands,
        positions=positions,
        fibers=fibers,
        ip_means=ip_means,
        pw_means=pw_means,
    )

    # z = np.load(args.out)
    # windows = z["features"]
    # labels = z["labels"]
    logging.info(f"Data shape: {windows.shape}; Labels shape: {labels.shape}")
    logging.info(
        f"Positives: {labels.sum():,}\tNegatives: {len(labels) - labels.sum():,}"
    )


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
    parser.add_argument("-c", "--ec", help="minimum ccs coverage", type=int, default=8)
    parser.add_argument("-s", "--sub-sample", type=float, default=1.0)
    parser.add_argument("-m", "--min-ml-score", type=int, default=200)
    parser.add_argument("--min-nuc-bp", type=int, default=2000)
    parser.add_argument("--min-nucs", type=int, default=10)
    parser.add_argument("--hifi", action="store_true")
    parser.add_argument(
        "--is_u16", help="Hifi kinetics are stored in u16, B,S", action="store_true"
    )
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    log_format = "[%(levelname)s][Time elapsed (ms) %(relativeCreated)d]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    if args.hifi:
        if args.is_u16:
            logging.debug("Using hifi with u16 (B,S) kinetics instead of u8 (B,C) data")
        data = make_hifi_kinetic_data(args.bam, args)
    else:
        fiber_data = read_fiber_data(
            args.all, args.bam, buffer=args.buffer, subsample=args.sub_sample
        )
        data = make_kinetic_data(args.bam, fiber_data, args)
        if args.out is not None:
            with open(args.out, "wb") as f:
                pickle.dump(data, f)


if __name__ == "__main__":
    main()
