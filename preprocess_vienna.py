"""
Flickr Vienna – cleaning & preprocessing pipeline
Stages:
  1. Load raw data
  2. Filter accuracy == "Street"
  3. Drop duplicate photo URLs
  4. Clip to Vienna administrative boundary (OSM)
  5. Dilute: keep one photo per (user, ~location, 30-min window)
  6. Classify users as tourist / local
"""

import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point

RAW_PATH = r"c:\projects\module_4\simulator\simulator_data\Vienna.txt"
OUT_PATH  = r"c:\projects\module_4\simulator\simulator_data\Vienna_clean.csv"

# ── helpers ──────────────────────────────────────────────────────────────────

def log(stage: str, df: pd.DataFrame, note: str = ""):
    msg = f"[{stage}] {len(df):>7,} rows"
    if note:
        msg += f"  |  {note}"
    print(msg)


# ── 1. LOAD ───────────────────────────────────────────────────────────────────

df = pd.read_csv(RAW_PATH)
log("LOAD", df, f"columns: {list(df.columns)}")

# Normalise accuracy capitalisation just in case
df["accuracy"] = df["accuracy"].str.strip()

# Parse datetime once
df["datetime"] = pd.to_datetime(df["datetime"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
n_bad_dt = df["datetime"].isna().sum()
if n_bad_dt:
    print(f"  WARNING: {n_bad_dt} rows with unparseable datetime - dropped")
    df = df.dropna(subset=["datetime"])

log("LOAD (after datetime parse)", df)

# ── 2. FILTER accuracy == "Street" ───────────────────────────────────────────

before = len(df)
df = df[df["accuracy"] == "Street"].copy()
log("ACCURACY FILTER", df, f"dropped {before - len(df):,} non-Street rows")

# ── 3. DEDUPLICATE photo URLs ─────────────────────────────────────────────────

before = len(df)
df = df.drop_duplicates(subset=["url"]).copy()
log("URL DEDUP", df, f"dropped {before - len(df):,} duplicate URLs")

# ── 4. CLIP TO VIENNA POLYGON (OSM) ──────────────────────────────────────────

print("\n[OSM BOUNDARY] fetching Vienna administrative boundary …")
vienna_gdf = ox.geocode_to_gdf("Vienna, Austria")
vienna_poly = vienna_gdf.geometry.iloc[0]
print(f"  Boundary bbox: {vienna_poly.bounds}")

geometry = [Point(xy) for xy in zip(df["long"], df["lat"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

before = len(gdf)
gdf = gdf[gdf.geometry.within(vienna_poly)].copy()
dropped = before - len(gdf)
log("SPATIAL CLIP", gdf, f"dropped {dropped:,} points outside Vienna polygon")

df = pd.DataFrame(gdf.drop(columns="geometry"))

# ── 5. DILUTE: one photo per (user, ~location, 30-min window) ─────────────────
#
# "~location": round coords to 4 decimal places (~11 m grid)
# "30-min window": floor datetime to 30-minute bins

df["lat_r"]   = df["lat"].round(4)
df["lon_r"]   = df["long"].round(4)
df["bin_30m"] = df["datetime"].dt.floor("30min")

before = len(df)
df = (
    df
    .sort_values("datetime")                          # keep earliest in each bin
    .drop_duplicates(subset=["user_id", "lat_r", "lon_r", "bin_30m"])
    .copy()
)
log("DILUTION (30-min / location bins)", df, f"dropped {before - len(df):,} burst duplicates")

df = df.drop(columns=["lat_r", "lon_r", "bin_30m"])

# ── 6. TOURIST / LOCAL CLASSIFICATION ────────────────────────────────────────
#
# Heuristic (no labels needed):
#   local   → photos span ≥ 2 distinct calendar years
#   tourist → all photos within a single calendar year
#
# Rationale: a local resident accumulates photos over multiple years;
# a tourist's visit is typically contained within one year.

user_years = df.groupby("user_id")["year"].nunique().rename("distinct_years")
df = df.merge(user_years, on="user_id", how="left")
df["is_tourist"] = (df["distinct_years"] == 1)
df = df.drop(columns=["distinct_years"])

tourist_users = df.groupby("user_id")["is_tourist"].first()
n_tourist = tourist_users.sum()
n_local   = (~tourist_users).sum()
log(
    "TOURIST/LOCAL",
    df,
    f"{n_local:,} local users | {n_tourist:,} tourist users "
    f"({n_tourist / (n_local + n_tourist) * 100:.1f}% tourist)"
)

# ── SAVE ──────────────────────────────────────────────────────────────────────

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}")
print(f"  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}")
