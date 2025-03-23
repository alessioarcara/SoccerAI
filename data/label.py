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
            # Skip challenge events
            if event_df.row(prev_idx, named=True)["possessionEventType"] == "CH":
                prev_idx -= 1
                continue

            pos_chain.append(prev_idx)
            prev_idx -= 1

        pos_chain = pos_chain[::-1]

        if chain_len is None or len(pos_chain) >= chain_len:
            pos_chains.append(pos_chain)

    return pos_chains


def neg_labeling(event_df: pl.DataFrame) -> List[List[int]]:
    pass

    # all_indices = list(range(event_data.height))
    # negative_indices = [index for index in all_indices if index not in indices]
    # tmp_indices = []
    # negative_indices_copy = negative_indices.copy()

    # for idx in range(len(negative_indices_copy) - 1):
    #     print(tmp_indices)
    #     tmp_indices.append(negative_indices_copy[idx])
    #     if not (negative_indices_copy[idx + 1] - negative_indices_copy[idx] == 1
    #         and event_data.row(negative_indices_copy[idx + 1], named=True)['teamName'] ==  event_data.row(negative_indices_copy[idx], named=True)['teamName']):
    #         if len(tmp_indices) < 5:
    #             negative_indices = [index for index in negative_indices if index not in tmp_indices]

    #         print(tmp_indices)
    #         tmp_indices = []
    #         print(tmp_indices)

    # event_data = event_data.with_columns(
    #     pl.when(pl.col("idx").is_in(indices)).then(1)
    #     .when(pl.col("idx").is_in(negative_indices)).then(0)
    #     .alias("label")
    # )
    # event_data = event_data.filter(pl.col("label").is_not_null())
