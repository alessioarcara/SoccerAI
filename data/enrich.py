import polars as pl
from fbref_utils import create_webdriver, get_player_id, get_player_html, extract_shoot_stats, extract_player_metastats
from transfermarkt_utils import search_player_id, get_avg_market_value_pre2021, get_player_physical_profile, find_best_player_match
from transfermarkt_utils import search_club_id, get_players_from_tmrooster
from utils import normalize
from config import SHOOTING_STATS, TEAM_ABBREVS


def enrich_player_record(player: dict, transfermarkt_id: str = None) -> dict:
    player_name = player["playerNickname"]
    player_team = player["playerTeam"]
    player_role = player["playerRole"]

    fbref_id = get_player_id(driver, player_name, player_team)
    player_html = get_player_html(driver, player_name, fbref_id)
    fbref_metastats = extract_player_metastats(player_html)

    if player_role == "GK":
        shooting_stats = {key: 0 for key in SHOOTING_STATS}
    else:
        shooting_stats = extract_shoot_stats(player_html, num_years_back=3)

    player_info = fbref_metastats | shooting_stats
    
    if transfermarkt_id is None:
        transfermarkt_id = search_player_id(player_name, player_team)
        
        
    if transfermarkt_id:
        transfermarkt_url = f"http://localhost:8000/players/{transfermarkt_id}/market_value"
        print(transfermarkt_url)
            
        market_value = get_avg_market_value_pre2021(transfermarkt_url)
        player_info["Market Value"] = market_value
        transfer_market_stats = get_player_physical_profile(transfermarkt_id)
        print(transfer_market_stats)
        for k, v in transfer_market_stats.items():
            player_info[k] = v
        
        player_info["tm_data_found"] = True
            
    else:
        player_info["tm_data_found"] = False

    player.update(player_info)
    return player


def enrich_roosters(rooster_df: pl.DataFrame) -> pl.DataFrame:
    teams_to_enrich = list(TEAM_ABBREVS.keys())
    all_updated_rows = [] 
    global driver  
    driver = create_webdriver()
    
    for team in teams_to_enrich:
        rooster_team_df = rooster_df.filter(pl.col("playerTeam") == team)
        enriched_team_rooster = enrich_team_rooster(rooster_team_df)
        enriched_team_rooster = enriched_team_rooster.drop("tm_data_found")
        enriched_team_rooster.write_csv(f'teams/{team}.csv')
        all_updated_rows.extend(enriched_team_rooster.to_dicts())
    
    enriched_roosters = pl.DataFrame(all_updated_rows)
    enriched_roosters.write_csv('enrich_roosters_final.csv')
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
    team_name = team_df.select('playerTeam').to_series()[0]
    
    # Check if any player has missing Transfermarkt data
    missing_df = team_df.filter(pl.col("tm_data_found") == False)
    if missing_df.height > 0:
        print(f"Found {missing_df.height} players with missing TM data for team {team_name}")
        team_df = fix_missing_tm_names_for_team(team_df, team_name)
    else:
        print("All players have Transfermarkt data.")

    return team_df




def fix_missing_tm_names_for_team(team_df: pl.DataFrame, team_name: str, season_id: int = 2022) -> pl.DataFrame:
    team_id = search_club_id(team_name)
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
                transfermarkt_id =  best_match["id"]
                row = enrich_player_record(row,transfermarkt_id)
            else:
                print(f"No match found for '{input_name}' in team {team_name}.")
                
                
        updated_rows.append(row)
        
    updated_df = pl.DataFrame(updated_rows)    
    return updated_df