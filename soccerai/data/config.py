from importlib.resources import as_file, files


# ================= RESOURCE PATHS =================
def get_resource_path(filename: str, package: str = "soccerai.data.resources") -> str:
    resource = files(package).joinpath(filename)
    with as_file(resource) as path:
        return str(path)


BALL_IMAGE_PATH = get_resource_path("soccer_ball_transparent.png")
PLAYER_STATS_PATH = get_resource_path("rosters.csv")
ACCEPTED_POS_CHAINS_PATH = get_resource_path("accepted_pos_chains.json")
ACCEPTED_NEG_CHAINS_PATH = get_resource_path("accepted_neg_chains.json")
DATASET_PATH = get_resource_path(
    "dataset.parquet", package="soccerai.data.resources.raw"
)

# ================= PITCH SETTINGS =================
PITCH_FIGSIZE = figsize = (12, 6.75)
PITCH_SETTINGS = {
    "pitch_type": "custom",
    "pitch_width": 68,
    "pitch_length": 105,
    "pitch_color": "grass",
    "line_color": "#c7d5cc",
    "stripe": True,
    "axis": True,
    "label": True,
}

# ================= BALL SETTINGS =================
BALL_ZOOM = 0.05
BALL_OFFSET_X = 1
BALL_OFFSET_Y = 1

# ================= UI LAYOUTS =================
# Button styling
BUTTON_STYLE = {
    "width": "200px",
    "height": "40px",
    "margin": "5px",
    "font_weight": "bold",
    "font_size": "14px",
}

# Layout for output widgets
PITCH_OUTPUT_LAYOUT = {
    "width": "45%",
    "border": "2px solid #ccc",
    "padding": "10px",
    "margin": "0 5px 0 0",
}

VIDEO_OUTPUT_LAYOUT = {
    "width": "55%",
    "border": "2px solid #ccc",
    "padding": "10px",
    "margin": "0 0 0 5px",
}

# Layout for Play widget and slider
PLAY_LAYOUT = {"width": "120px", "margin": "0 10px 0 20px"}

SLIDER_LAYOUT = {"width": "300px", "margin": "10px 10px 10px 20px"}

# Layout for boxes
TOP_BOX_LAYOUT = {
    "justify_content": "center",
    "align_items": "stretch",
    "margin": "20px 0 0 0",
}

CONTROLS_BOX_LAYOUT = {
    "justify_content": "center",
    "align_items": "center",
    "margin": "10px 0 10px 0",
}

BUTTONS_BOX_LAYOUT = {
    "justify_content": "center",
    "align_items": "center",
    "margin": "10px 0 20px 0",
}

TEAM_ABBREVS = {
    "Qatar": "qat",
    "Ecuador": "ecu",
    "Senegal": "sen",
    "Netherlands": "ned",
    "England": "eng",
    "Iran": "irn",
    "United States": "usa",
    "Argentina": "arg",
    "Saudi Arabia": "ksa",
    "Mexico": "mex",
    "Poland": "pol",
    "France": "fra",
    "Australia": "aus",
    "Denmark": "den",
    "Tunisia": "tun",
    "Spain": "esp",
    "Costa Rica": "crc",
    "Germany": "ger",
    "Japan": "jpn",
    "Belgium": "bel",
    "Canada": "can",
    "Morocco": "mar",
    "Croatia": "cro",
    "Brazil": "bra",
    "Serbia": "srb",
    "Switzerland": "sui",
    "Cameroon": "cmr",
    "Portugal": "por",
    "Ghana": "gha",
    "Uruguay": "uru",
    "South Korea": "kor",
    "Wales": "wal",
}

SHOOTING_STATS = [
    "goals",  # Gls
    "shots",  # Sh
    "shots_on_target",  # SoT
    "shots_on_target_pct",  # SoT%
    "shots_per90",  # Sh/90
    "shots_on_target_per90",  # SoT/90
    "goals_per_shot",  # G/Sh
    "goals_per_shot_on_target",  # G/SoT
    "average_shot_distance",  # Dist
    "pens_made",  # PK
    "pens_att",  # PKatt
]
# ================= GOAL COORDINATES =================
X_GOAL_RIGHT = 105.0
X_GOAL_LEFT = 0.0
Y_GOAL = 34.0
