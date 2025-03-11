from typing import List, Optional
import polars as pl


def pos_labeling(
    event_df: pl.DataFrame, chain_len: Optional[int] = None
) -> List[List[int]]:
    """ """
    shots_df = event_df.filter(event_df["possessionEventType"] == "SH")
    pos_chains = []

    for shot in shots_df.iter_rows(named=True):
        shot_idx = shot["index"]
        team_name = shot["teamName"]

        pos_chain = [shot_idx]
        prev_idx = shot_idx - 1

        while (
            prev_idx >= 0
            and event_df.row(prev_idx, named=True)["teamName"] == team_name
        ):
            pos_chain.append(prev_idx)
            prev_idx -= 1

        pos_chain = pos_chain[::-1]

        if chain_len is None or len(pos_chain) >= chain_len:
            pos_chains.append(pos_chain)

    return pos_chains


def neg_labeling(event_df: pl.DataFrame) -> List[List[int]]:
    pass
