"""Persist annotations using event identities, never loader row positions."""

from itertools import pairwise
from typing import Any

import polars as pl

FRAME_KEYS = ["gameId", "gameEventId", "possessionEventId"]
ANNOTATION_VERSION = 2


def encode_chains(chains: list[list[int]], events: pl.DataFrame) -> dict[str, Any]:
    rows = events.select("index", *FRAME_KEYS).unique().to_dicts()
    lookup = {row["index"]: [row[c] for c in FRAME_KEYS] for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("An event index identifies multiple events")
    encoded = []
    for chain in chains:
        if not chain or len(set(chain)) != len(chain):
            raise ValueError(
                "Annotations must contain nonempty chains without repeated frames"
            )
        try:
            encoded.append([lookup[idx] for idx in chain])
        except KeyError as exc:
            raise ValueError(f"Unknown annotated event index: {exc.args[0]}") from exc
    return {"version": ANNOTATION_VERSION, "event_key": FRAME_KEYS, "chains": encoded}


def decode_chains(payload: Any, events: pl.DataFrame) -> list[list[int]]:
    if type(payload) is list:
        raise ValueError(
            "Legacy row-index annotations must be migrated with "
            "scripts/repair_dataset_annotations.py and the original parquet"
        )
    if not isinstance(payload, dict) or payload.get("version") != ANNOTATION_VERSION:
        raise ValueError("Unsupported annotation format")
    if payload.get("event_key") != FRAME_KEYS or not isinstance(
        payload.get("chains"), list
    ):
        raise ValueError("Invalid annotation event keys or chains")
    lookup: dict[tuple, int] = {}
    ambiguous = set()
    for row in events.select("index", *FRAME_KEYS).unique().iter_rows(named=True):
        key = tuple(row[c] for c in FRAME_KEYS)
        if key in lookup and lookup[key] != row["index"]:
            ambiguous.add(key)
        lookup[key] = row["index"]
    decoded = []
    for chain in payload["chains"]:
        if not isinstance(chain, list) or not chain:
            raise ValueError("Annotations contain an empty or invalid chain")
        indices = []
        for identity in chain:
            if not isinstance(identity, list) or len(identity) != len(FRAME_KEYS):
                raise ValueError(f"Invalid annotated event identity: {identity!r}")
            key = tuple(identity)
            if key in ambiguous:
                raise ValueError(f"Ambiguous annotated event identity: {key}")
            if key not in lookup:
                raise ValueError(f"Annotated event is missing from the raw data: {key}")
            indices.append(lookup[key])
        if len(set(indices)) != len(indices):
            raise ValueError("Annotations contain repeated frames within a chain")
        decoded.append(indices)
    return decoded


def chain_errors(
    chains: list[list[int]], events: pl.DataFrame, *, positive: bool
) -> dict[int, list[str]]:
    """Validate against the complete timeline, including unselected events."""
    rows = events.sort("index").to_dicts()
    positions = {row["index"]: pos for pos, row in enumerate(rows)}
    errors = {}
    for cid, chain in enumerate(chains):
        reasons = []
        if not chain or any(idx not in positions for idx in chain):
            errors[cid] = ["empty chain or missing event"]
            continue
        selected = [rows[positions[idx]] for idx in chain]
        if any(a >= b for a, b in pairwise(chain)):
            reasons.append("unordered or repeated events")
        for field in ["gameId", "period", "teamName"]:
            if len({r[field] for r in selected}) != 1:
                reasons.append(f"multiple {field} values")
        if selected[0]["teamName"] is None:
            reasons.append("unknown possession team")
        if any(r["period"] not in (1, 2, 3, 4) for r in selected):
            reasons.append("invalid period")
        if any(r["possessionEventType"] is None for r in selected):
            reasons.append("missing possession event")
        if positive:
            if selected[-1]["possessionEventType"] != "SH":
                reasons.append("positive chain has no terminal shot")
        elif any(r["possessionEventType"] == "SH" for r in selected):
            reasons.append("negative chain contains a shot")
        start, end = min(positions[i] for i in chain), max(positions[i] for i in chain)
        if any(
            r["gameId"] != selected[0]["gameId"]
            or r["period"] != selected[0]["period"]
            or r["teamName"] not in (selected[0]["teamName"], None)
            for r in rows[start : end + 1]
        ):
            reasons.append("chain skips a possession or period boundary")
        if reasons:
            errors[cid] = reasons
    return errors
