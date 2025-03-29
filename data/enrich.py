import polars as pl
from fbref_utils import create_webdriver, get_player_id, get_player_html, extract_shoot_stats, extract_player_metastats
from transfermarkt_utils import search_player_id, get_avg_market_value_pre2021, get_player_physical_profile
from config import SHOOTING_STATS


def enrich_player_record(player: dict) -> dict:
    player_name = player["playerNickname"]
    player_team = player["playerTeam"]
    player_role = player["playerRole"]

    print(player_name)
    fbref_id = get_player_id(driver, player_name, player_team)
    print(fbref_id)
    player_html = get_player_html(driver, player_name, fbref_id)
    fbref_metastats = extract_player_metastats(player_html)
    print(f"Metastats added for {player_name}")

    if player_role == "GK":
        shooting_stats = {key: 0 for key in SHOOTING_STATS}
    else:
        shooting_stats = extract_shoot_stats(player_html, num_years_back=3, min_minute_90s=5)

    player_info = fbref_metastats | shooting_stats

    transfermarkt_id = search_player_id(player_name, player_team)
    if transfermarkt_id:
        print(transfermarkt_id)
        market_value = get_avg_market_value_pre2021(transfermarkt_id)
        player_info["Market Value"] = market_value
        transfer_market_stats = get_player_physical_profile(transfermarkt_id)
        print(transfer_market_stats)
        for k, v in transfer_market_stats.items():
            player_info[k] = v

    player.update(player_info)
    return player


def enrich_roosters(rooster_df: pl.DataFrame) -> pl.DataFrame:
    rooster_rows = rooster_df.to_dicts()
    updated_rows = []
    global driver  
    driver = create_webdriver()

    for player in rooster_rows:
        enriched_player = enrich_player_record(player)
        updated_rows.append(enriched_player)

    enriched_roosters = pl.DataFrame(updated_rows)
    enriched_roosters.write_csv('enrich_roosters.csv')
    driver.quit()
    return enriched_roosters



