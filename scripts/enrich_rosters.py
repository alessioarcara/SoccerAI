#!/usr/bin/env python3

import argparse

from loguru import logger

from soccerai.data.data import load_and_process_rosters
from soccerai.data.enrichers.rosters import RostersEnricher


def main():
    parser = argparse.ArgumentParser(
        description="Script for rosters enrichment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", required=True, help="Server host address")
    parser.add_argument("--port", required=True, help="Server port")

    args = parser.parse_args()

    roster_df = load_and_process_rosters("/home/soccerdata/FIFA_WorldCup_2022/Rosters")

    enricher = RostersEnricher(roster_df, f"http://{args.host}:{args.port}")
    try:
        enricher()
    except Exception as e:
        logger.error("Error during enrichment: {}", str(e))
        exit(1)

    print("Enrichment complete. Output saved to enriched_rosters.csv")


if __name__ == "__main__":
    main()
