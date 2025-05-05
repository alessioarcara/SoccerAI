from typing import Dict, Optional

import polars as pl
from loguru import logger

import soccerai.data.scraping.fbref as fbref
import soccerai.data.scraping.transfermarkt as tm
from soccerai.data.config import SHOOTING_STATS, TEAM_ABBREVS
from soccerai.data.scraping.utils import create_webdriver, normalize


class RoostersEnricher:
    def __init__(
        self, rooster_df: pl.DataFrame, base_url: str = "http://localhost:8000"
    ):
        self._driver = create_webdriver()
        self.rooster_df = rooster_df
        self._base_url = base_url

    def _enrich_player_record(
        self, player: Dict, tm_player_id: Optional[str] = None
    ) -> Dict:
        player_name = player["playerNickname"]
        player_team = player["playerTeam"]
        player_role = player["playerRole"]
        combined_stats: Dict = {}

        if tm_player_id is None:
            tm_player_id = tm.search_player_id(player_name, player_team, self._base_url)

        if tm_player_id:
            tm_url = f"{self._base_url}/players/{tm_player_id}/market_value"
            tm_market_value = tm.get_avg_market_value_pre2021(tm_url)
            combined_stats["Market Value"] = tm_market_value

            tm_profile = tm.get_player_physical_profile(tm_player_id, self._base_url)
            if tm_profile is None:
                tm_profile = {}
            for k, v in tm_profile.items():
                combined_stats[k] = v
            combined_stats["tm_data_found"] = True
        else:
            combined_stats["tm_data_found"] = False

        fbref_id = fbref.get_player_id(self._driver, player_name, player_team)
        fbref_html = fbref.get_html(self._driver, player_name, fbref_id)
        fbref_stats = fbref.extract_metastats(fbref_html)
        logger.debug(fbref_stats)

        if player_role == "GK":
            fbref_shoot_stats = {key: 0 for key in SHOOTING_STATS}
        else:
            fbref_shoot_stats = fbref.extract_shoot_stats(fbref_html, num_years_back=3)

        combined_stats = fbref_stats | fbref_shoot_stats | combined_stats

        player.update(combined_stats)
        return player

    def __call__(self):
        all_updated_rows = []
        for team_name in list(TEAM_ABBREVS.keys()):
            rooster_team_df = self.rooster_df.filter(pl.col("playerTeam") == team_name)
            enriched_team_rooster = self._enrich_team_rooster(rooster_team_df)
            enriched_team_rooster = enriched_team_rooster.drop("tm_data_found")
            enriched_team_rooster.write_csv(f"teams/{team_name}.csv")
            all_updated_rows.extend(enriched_team_rooster.to_dicts())
        enriched_roosters = pl.DataFrame(all_updated_rows)
        enriched_roosters.write_csv("enrich_roosters_final.csv")
        self._driver.quit()

    def _enrich_team_rooster(self, rooster_team_df: pl.DataFrame) -> pl.DataFrame:
        # Enrichment initial for every player in the team
        team_rows = rooster_team_df.to_dicts()
        updated_rows = []
        for player in team_rows:
            enriched_player = self._enrich_player_record(player)
            updated_rows.append(enriched_player)
        team_df = pl.DataFrame(updated_rows)
        team_name = team_df.select("playerTeam").to_series()[0]
        # Check if any player has missing Transfermarkt data
        missing_df = team_df.filter(pl.col("tm_data_found") is False)
        if missing_df.height > 0:
            logger.warning(
                f"Found {missing_df.height} players with missing TM data for team {team_name}"
            )
            team_df = self._fix_missing_tm_names_for_team(team_df, team_name)
        else:
            logger.info("All players have Transfermarkt data.")
        return team_df

    def _fix_missing_tm_names_for_team(
        self, team_df: pl.DataFrame, team_name: str, season_id: int = 2022
    ) -> pl.DataFrame:
        team_id = tm.search_club_id(team_name, self._base_url)
        if team_id is None:
            logger.warning(
                f"Team ID not found for team {team_name}. Skipping name correction."
            )
            return team_df

        team_slug = normalize(team_name).replace(" ", "-")
        rooster_url = f"https://www.transfermarkt.it/{team_slug}/kader/verein/{team_id}/plus/0/galerie/0?saison_id={season_id}"
        logger.info(f"Roster URL for team {team_name}: {rooster_url}")

        # Scrape the roster page to get the list of players
        roster_list = tm.get_players_from_rooster(rooster_url)
        logger.debug(
            f"Found {len(roster_list)} players in roster for team {team_name}."
        )

        updated_rows = []
        for row in team_df.to_dicts():
            if not row.get("tm_data_found", False):
                input_name = row["playerNickname"]
                best_match = tm.find_best_player_match(input_name, roster_list)
                if best_match:
                    logger.debug(f"Correcting '{input_name}' to '{best_match['name']}'")
                    row["playerNickname"] = best_match["name"]
                    transfermarkt_id = best_match["id"]
                    row = self._enrich_player_record(row, transfermarkt_id)
                else:
                    logger.warning(
                        f"No match found for '{input_name}' in team {team_name}."
                    )
            updated_rows.append(row)
        updated_df = pl.DataFrame(updated_rows)
        return updated_df
