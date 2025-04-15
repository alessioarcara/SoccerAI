#!/usr/bin/env python3

import argparse

from loguru import logger

from soccerai.data.data import load_and_process_roosters
from soccerai.data.scraping.enrich import RoostersEnricher


def main():
    parser = argparse.ArgumentParser(
        description="Script for roosters enrichment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", required=True, help="Server host address")
    parser.add_argument("--port", required=True, help="Server port")

    args = parser.parse_args()

    rooster_df = load_and_process_roosters(
        "/home/soccerdata/FIFA_WorldCup_2022/Rosters"
    )

    enricher = RoostersEnricher(rooster_df, f"http://{args.host}:{args.port}")
    try:
        enricher()
    except Exception as e:
        logger.error("Error during enrichment: {}", str(e))
        exit(1)

    print("Enrichment complete. Output saved to enriched_roosters.csv")


if __name__ == "__main__":
    main()
