from loguru import logger
from tqdm import tqdm

from soccerai.data.data import (
    load_and_process_metadata,
    load_and_process_rosters,
    load_and_process_soccer_events,
)
from soccerai.data.label import get_chains
from soccerai.data.utils import download_video_frames

if __name__ == "__main__":
    event_df, players_df = load_and_process_soccer_events(
        "/home/soccerdata/FIFA_WorldCup_2022/Event Data"
    )
    metadata_df = load_and_process_metadata(
        "/home/soccerdata/FIFA_WorldCup_2022/Metadata"
    )
    rosters_df = load_and_process_rosters("/home/soccerdata/FIFA_WorldCup_2022/Rosters")

    chains_dict = get_chains(
        event_df,
        players_df,
        metadata_df,
        rosters_df,
        skip_challenge_events=False,
        use_player_pos=True,
    )

    all_chains = chains_dict["all_pos_chains"] + chains_dict["all_neg_chains"]

    logger.info(f"Downloading video frames for {len(all_chains)} chains...")

    for chain in tqdm(all_chains, desc="Downloading"):
        try:
            download_video_frames(chain, event_df, "/home/soccerdata/frames")
        except Exception as e:
            logger.error(f"Error downloading frames for a chain: {e}")

    logger.success("All frames downloaded.")
