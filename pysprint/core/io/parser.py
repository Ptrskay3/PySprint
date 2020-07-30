import re
from pathlib import Path
from typing import IO, AnyStr, List, Optional, Union
import warnings

import pandas as pd

from pysprint.utils.exceptions import PySprintWarning

PathOrBuffer = Union[str, Path, IO[AnyStr]]


def _parse_single_line(line: str):
    possible_separators = (":", "_", "-", "\t", " ", ";", ",", ".")
    idx = 0
    parsed_line = (line,)
    while idx < len(possible_separators):
        res = line.split(possible_separators[idx])
        if len(res) == 2:
            parsed_line = res
            break
        idx += 1

    # if we fail with the possible separators, let's see if
    # we can split up on contiguous whitespace to two pieces.

    if len(parsed_line) == 1:
        ln = re.split(r'\s+', *parsed_line)
        if len(ln) == 2:
            parsed_line = ln
    rx = '[' + re.escape(''.join(possible_separators[2:-4])) + ']'

    # remove the obvious unwanted chars
    parsed_line = [element.strip("\n").strip("\x00") for element in parsed_line]

    # remove the separator-like chars
    parsed_line = [re.sub(rx, "", el) for el in parsed_line]

    # remove whitespace around elements
    parsed_line = [element.strip(" ") for element in parsed_line]

    return parsed_line


def _parse_metadata(filename, ref=None, sam=None, meta_len=1, encoding='utf-8'):
    _meta = {}
    with open(filename, encoding=encoding) as file:
        comm = next(file).strip("\n").split("-")[-1].lstrip(" ")
        additional = [
            _parse_single_line(next(file))
            for _ in range(1, meta_len + 1)
        ]
        if meta_len != 0:
            _meta = {"comment": comm}
        for el in additional:
            if len(el) == 2:
                _meta[el[0]] = el[1]
            else:
                if "unparsed" not in _meta:
                    _meta["unparsed"] = []
                _meta["unparsed"].append(el)
    if ref is not None:
        with open(ref, encoding=encoding) as file:
            comm = next(file).strip("\n").split("-")[-1].lstrip(" ")
            if meta_len != 0:
                _meta["reference_comment"] = comm
    if sam is not None:
        with open(sam, encoding=encoding) as file:
            comm = next(file).strip("\n").split("-")[-1].lstrip(" ")
            if meta_len != 0:
                _meta["sample_comment"] = comm
    return _meta


def _parse_raw(
        filename: PathOrBuffer,
        ref: Optional[PathOrBuffer] = None,
        sam: Optional[PathOrBuffer] = None,
        skiprows: int = 0,
        decimal: str = ".",
        sep: Optional[str] = None,
        delimiter: Optional[str] = None,
        comment: Optional[str] = None,
        usecols: Optional[List] = None,
        names=None,
        swapaxes=False,
        na_values=None,
        skip_blank_lines=True,
        keep_default_na=False,
        meta_len=1
):
    if len([_ for _ in (ref, sam) if _ is not None]) == 1:
        warnings.warn(
            "Reference and sample arm should be passed together or neither one.",
            PySprintWarning
        )
    if sep and delimiter:
        warnings.warn("Specified both `sep` and `delimiter`, using `delimiter` as default.")
        separator = delimiter
    else:
        separator = sep or delimiter or ","

    if comment is None:
        comment = "#"

    if usecols is None:
        usecols = [0, 1]

    if names is None:
        names = ["x", "y"]

    if skiprows < meta_len:
        warnings.warn(
            f"Skiprows is currently {skiprows}, but"
            f" meta information is set to {meta_len}"
            " lines. This implies that either one is probably wrong.",
            PySprintWarning,
        )

    _meta = _parse_metadata(filename, ref, sam, meta_len=meta_len)

    df = pd.read_csv(
        filename,
        skiprows=skiprows,
        sep=separator,
        decimal=decimal,
        usecols=usecols,
        names=names,
        comment=comment,
        na_values=na_values,
        skip_blank_lines=skip_blank_lines,
        keep_default_na=keep_default_na
    )
    if ref is not None and sam is not None:
        r = pd.read_csv(
            ref,
            skiprows=skiprows,
            sep=separator,
            decimal=decimal,
            usecols=usecols,
            names=names,
            comment=comment,
            na_values=na_values,
            skip_blank_lines=skip_blank_lines,
            keep_default_na=keep_default_na
        )
        s = pd.read_csv(
            sam,
            skiprows=skiprows,
            sep=separator,
            decimal=decimal,
            usecols=usecols,
            names=names,
            comment=comment,
            na_values=na_values,
            skip_blank_lines=skip_blank_lines,
            keep_default_na=keep_default_na
        )

        if swapaxes:
            return {
                "x": df["y"].values,
                "y": df["x"].values,
                "ref": r["x"].values,
                "sam": s["x"].values,
                "meta": _meta
            }
        return {
            "x": df["x"].values,
            "y": df["y"].values,
            "ref": r["y"].values,
            "sam": s["y"].values,
            "meta": _meta
        }
    if swapaxes:
        return {
            "x": df["y"].values,
            "y": df["x"].values,
            "meta": _meta
        }
    return {
        "x": df["x"].values,
        "y": df["y"].values,
        "meta": _meta
    }
