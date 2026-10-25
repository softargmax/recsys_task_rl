from dataclasses import dataclass

DAY = 86400
SESSION_IDLE = 30 * 60
LISTEN_PLUS = 50

TEST_DAYS = 7
VAL_DAYS = 7
RL_DAYS = 14
QUERIES_PER_USER = 10

CATALOG_SIZE = 50_000
ARTIST_VOCAB = 20_000

PAD = 0
UNK = 1
N_SPECIAL = 2

RATIO_EDGES = (5, 10, 25, 50, 75, 95)
DUR_EDGES = (60, 120, 150, 180, 210, 240, 300, 420, 600)
GAP_EDGES = (10, 60, 300, SESSION_IDLE, 6 * 3600, DAY, 7 * DAY)
MATCH_TAUS = (3600, DAY, 30 * DAY)
MATCH_WINDOW = 4096

N_RATIO = len(RATIO_EDGES) + 2
N_DUR = len(DUR_EDGES) + 2
N_GAP = len(GAP_EDGES) + 2
GAP_GENERATED = 1

CONT_UNKNOWN = 2
MAX_SLATE = 8

EVENT_FIELDS = ("item", "artist", "ts", "ratio", "organic", "dur_b", "gap_b", "ratio_b", "cont")


@dataclass
class ModelConfig:
    d: int = 256
    layers: int = 4
    heads: int = 4
    ctx: int = 256
    dropout: float = 0.1


MODEL_SIZES = {
    "tiny": ModelConfig(d=64, layers=2, heads=2, ctx=64),
    "small": ModelConfig(d=128, layers=2, heads=4, ctx=256),
    "base": ModelConfig(d=256, layers=4, heads=4, ctx=256),
    "large": ModelConfig(d=512, layers=8, heads=8, ctx=256),
}
