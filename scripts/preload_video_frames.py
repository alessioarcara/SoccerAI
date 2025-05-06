from loguru import logger
from tqdm import tqdm

from soccerai.data.data import load_and_process_metadata, load_and_process_soccer_events
from soccerai.data.label import neg_labeling, pos_labeling
from soccerai.data.utils import download_video_frames

if __name__ == "__main__":
    event_df, players_df = load_and_process_soccer_events(
        "/home/soccerdata/FIFA_WorldCup_2022/Event Data"
    )
    metadata_df = load_and_process_metadata(
        "/home/soccerdata/FIFA_WorldCup_2022/Metadata"
    )

    chain_len = 1
    pos_chains = pos_labeling(event_df, chain_len)
    logger.info(f"Generated {len(pos_chains)} positive chains")

    neg_chains = neg_labeling(event_df, players_df, metadata_df, chain_len, 25.0)
    logger.info(f"Generated {len(neg_chains)} negative chains")

    all_chains = pos_chains + neg_chains

    logger.info(f"Downloading video frames for {len(all_chains)} chains...")

    for chain in tqdm(all_chains, desc="Downloading"):
        try:
            download_video_frames(chain, event_df, "/home/soccerdata/frames")
        except Exception as e:
            logger.error(f"Error downloading frames for a chain: {e}")

    logger.success("All frames downloaded.")
