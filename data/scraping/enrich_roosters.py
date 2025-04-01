from data.data import load_and_process_roosters
from data.scraping.enrich import enrich_roosters


def main():
    rooster_df = load_and_process_roosters(
        "/home/soccerdata/FIFA_WorldCup_2022/Rosters"
    )
    enrich_roosters(rooster_df)
    print("Enrichment complete. Output saved to enriched_roosters.csv")


if __name__ == "__main__":
    main()
