import sys

from loguru import logger

from data.data import load_and_process_roosters
from data.scraping.enrich import RoostersEnricher


def main():
    if len(sys.argv) < 3:
        logger.error("Parameters not correct.")
        sys.exit(1)

    rooster_df = load_and_process_roosters(
        "/home/soccerdata/FIFA_WorldCup_2022/Rosters"
    )
    enricher = RoostersEnricher(rooster_df, f"{sys.argv[1]}:{sys.argv[2]}")
    enricher()
    print("Enrichment complete. Output saved to enriched_roosters.csv")


if __name__ == "__main__":
    main()
