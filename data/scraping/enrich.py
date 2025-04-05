import polars as pl

from data.config import SHOOTING_STATS, TEAM_ABBREVS
from data.scraping.fbref import (
    create_webdriver,
    extract_fbref_metastats,
    extract_fbref_shoot_stats,
    get_fbref_html,
    get_fbref_player_id,
)
from data.scraping.transfermarkt import (
    find_best_player_match,
    get_avg_market_value_pre2021,
    get_player_physical_profile,
    get_players_from_tmrooster,
    search_tm_club_id,
    search_tm_player_id,
)
from data.scraping.utils import normalize


def enrich_player_record(player: dict, tm_player_id: str = None) -> dict:
    player_name = player["playerNickname"]
    player_team = player["playerTeam"]
    player_role = player["playerRole"]
    combined_stats = {}

    if tm_player_id is None:
        tm_player_id = search_tm_player_id(player_name, player_team)

    if tm_player_id:
        tm_url = f"http://localhost:8000/players/{tm_player_id}/market_value"
        tm_market_value = get_avg_market_value_pre2021(tm_url)
        combined_stats["Market Value"] = tm_market_value

        tm_profile = get_player_physical_profile(tm_player_id)
        if tm_profile is None:
            tm_profile = {}
        for k, v in tm_profile.items():
            combined_stats[k] = v
        combined_stats["tm_data_found"] = True
    else:
        combined_stats["tm_data_found"] = False

    fbref_id = get_fbref_player_id(driver, player_name, player_team)
    fbref_html = get_fbref_html(driver, player_name, fbref_id)
    fbref_stats = extract_fbref_metastats(fbref_html)
    print(fbref_stats)

    if player_role == "GK":
        fbref_shoot_stats = {key: 0 for key in SHOOTING_STATS}
    else:
        fbref_shoot_stats = extract_fbref_shoot_stats(fbref_html, num_years_back=3)

    combined_stats = fbref_stats | fbref_shoot_stats | combined_stats

    player.update(combined_stats)
    return player


def enrich_roosters(rooster_df: pl.DataFrame) -> pl.DataFrame:
    all_updated_rows = []
    global driver
    driver = create_webdriver()
    for team_name in list(TEAM_ABBREVS.keys()):
        rooster_team_df = rooster_df.filter(pl.col("playerTeam") == team_name)
        enriched_team_rooster = enrich_team_rooster(rooster_team_df)
        enriched_team_rooster = enriched_team_rooster.drop("tm_data_found")
        enriched_team_rooster.write_csv(f"teams/{team_name}.csv")
        all_updated_rows.extend(enriched_team_rooster.to_dicts())

    enriched_roosters = pl.DataFrame(all_updated_rows)
    enriched_roosters.write_csv("enrich_roosters_final.csv")
    driver.quit()
    return enriched_roosters


def enrich_team_rooster(rooster_team_df: pl.DataFrame) -> pl.DataFrame:
    # Enrichment initial for every player in the team
    team_rows = rooster_team_df.to_dicts()
    updated_rows = []
    for player in team_rows:
        enriched_player = enrich_player_record(player)
        updated_rows.append(enriched_player)
    team_df = pl.DataFrame(updated_rows)
    team_name = team_df.select("playerTeam").to_series()[0]

    # Check if any player has missing Transfermarkt data
    missing_df = team_df.filter(pl.col("tm_data_found") is False)
    if missing_df.height > 0:
        print(
            f"Found {missing_df.height} players with missing TM data for team {team_name}"
        )
        team_df = fix_missing_tm_names_for_team(team_df, team_name)
    else:
        print("All players have Transfermarkt data.")

    return team_df


def fix_missing_tm_names_for_team(
    team_df: pl.DataFrame, team_name: str, season_id: int = 2022
) -> pl.DataFrame:
    team_id = search_tm_club_id(team_name)
    if team_id is None:
        print(f"Team ID not found for team {team_name}. Skipping name correction.")
        return team_df

    team_slug = normalize(team_name).replace(" ", "-")
    rooster_url = f"https://www.transfermarkt.it/{team_slug}/kader/verein/{team_id}/plus/0/galerie/0?saison_id={season_id}"
    print(f"Roster URL for team {team_name}: {rooster_url}")

    # Scrape the roster page to get the list of players
    roster_list = get_players_from_tmrooster(rooster_url)
    print(f"Found {len(roster_list)} players in roster for team {team_name}.")

    updated_rows = []
    for row in team_df.to_dicts():
        if not row.get("tm_data_found", False):
            input_name = row["playerNickname"]
            best_match = find_best_player_match(input_name, roster_list)
            if best_match:
                print(f"Correcting '{input_name}' to '{best_match['name']}'")
                row["playerNickname"] = best_match["name"]
                # Re-run enrichment for this player using the corrected name
                transfermarkt_id = best_match["id"]
                row = enrich_player_record(row, transfermarkt_id)
            else:
                print(f"No match found for '{input_name}' in team {team_name}.")

        updated_rows.append(row)

    updated_df = pl.DataFrame(updated_rows)
    return updated_df
