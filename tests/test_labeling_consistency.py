import polars as pl
import pytest

from soccerai.data.data import (
    ACCEPTED_NEG_CHAINS_PATH,
    ACCEPTED_POS_CHAINS_PATH,
    _load_chains,
    load_and_process_soccer_events,
)

EVENT_DATA_PATH = "/home/soccerdata/FIFA_WorldCup_2022/Event Data"


@pytest.mark.parametrize(
    "chains_path,expect_shot",
    [(ACCEPTED_POS_CHAINS_PATH, True), (ACCEPTED_NEG_CHAINS_PATH, False)],
)
def test_labeling_consistenty(chains_path: str, expect_shot: bool):
    event_df, _ = load_and_process_soccer_events(
        EVENT_DATA_PATH, filter_invalid_events=True
    )
    chains = _load_chains(chains_path)

    for chain in chains:
        chain_df = event_df.filter(pl.col("index").is_in(chain))

        possessionEventTypes = chain_df["possessionEventType"].to_list()
        if expect_shot:
            assert "SH" in possessionEventTypes[-1]

        unique_team_names = set(chain_df["teamName"].to_list())
        assert len(unique_team_names) == 1
