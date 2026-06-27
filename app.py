"""
Flickr Vienna – preprocessing + POI explorer + spatial clustering
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

_DATA        = Path(__file__).parent / "simulator_data"
RAW_PATH     = _DATA / "Vienna.txt"
OUT_PATH     = _DATA / "Vienna_clean.csv"
CLUSTER_PATH = _DATA / "Vienna_clustered.csv"
INNERE_STADT = (48.2093, 16.3728)

FEATURE_COLS = [
    "n_locations", "time_span_hours",
    "radius_of_gyration_km", "path_linearity", "median_step_km",
    "location_entropy", "pct_center_locations",
    "pct_tourism", "pct_food", "pct_local", "pct_leisure",
]

FEAT_LABELS = {
    "n_locations":              ("many visit events",                "few visit events"),
    "time_span_hours":       ("long day (many hours active)",     "short day (few hours)"),
    "radius_of_gyration_km": ("wide spatial spread",              "narrow spread"),
    "path_linearity":        ("linear A→B route",                "circular / area loop"),
    "median_step_km":        ("large landmark-to-landmark jumps", "slow local drift"),
    "location_entropy":      ("spread across many spots",         "concentrated at few spots"),
    "pct_center_locations":     ("city-centre focused",              "exploring outer districts"),
    "pct_tourism":           ("near tourist POIs",                "avoids tourist spots"),
    "pct_food":              ("food & drink oriented",            "avoids food venues"),
    "pct_local":             ("local services area",              "avoids local services"),
    "pct_leisure":           ("leisure & parks",                  "avoids leisure areas"),
}

CLUSTER_COLORS = [
    "#E6194B",  # red
    "#4363D8",  # blue
    "#3CB44B",  # green
    "#F58231",  # orange
    "#911EB4",  # purple
    "#42D4F4",  # cyan
    "#F032E6",  # magenta
    "#FFE119",  # yellow
    "#469990",  # teal
    "#DCBEFF",  # lavender
]


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float))
                               for x in [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _entropy(labels):
    if len(labels) <= 1:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p + 1e-12)).sum())


st.set_page_config(page_title="Vienna Tourist Simulator - Or Gur", layout="wide")
st.title("Vienna Tourist Simulator – Preprocessing & Spatial Analysis")
st.markdown(
    "A 5-stage cleaning pipeline, tourist/local classification, "
    "POI coverage explorer, and unsupervised behavioural clustering."
)

# ════════════════════════════════════════════════════════════════════════════════
# 1. PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Fetching Vienna boundary from OSM...")
def get_vienna_polygon():
    gdf = ox.geocode_to_gdf("Vienna, Austria")
    return gdf.geometry.iloc[0]


@st.cache_data(show_spinner="Running preprocessing pipeline...")
def run_pipeline():
    stages = []

    df = pd.read_csv(RAW_PATH)
    df["accuracy"] = df["accuracy"].str.strip()
    df["datetime"] = pd.to_datetime(df["datetime"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    bad_dt = int(df["datetime"].isna().sum())
    df = df.dropna(subset=["datetime"])
    stages.append(("Load", len(df), bad_dt, "bad datetime rows dropped"))

    before = len(df)
    df = df[df["accuracy"] == "Street"].copy()
    stages.append(("Accuracy = Street", len(df), before - len(df), "non-Street rows dropped"))

    before = len(df)
    df = df.drop_duplicates(subset=["url"]).copy()
    stages.append(("URL dedup", len(df), before - len(df), "duplicate URLs dropped"))

    vienna_poly = get_vienna_polygon()
    geom = [Point(xy) for xy in zip(df["long"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
    before = len(gdf)
    gdf = gdf[gdf.geometry.within(vienna_poly)].copy()
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    stages.append(("Spatial clip (OSM)", len(df), before - len(df), "points outside Vienna dropped"))

    df["lat_r"]   = df["lat"].round(4)
    df["lon_r"]   = df["long"].round(4)
    df["bin_30m"] = df["datetime"].dt.floor("30min")
    before = len(df)
    df = (df.sort_values("datetime")
            .drop_duplicates(subset=["user_id", "lat_r", "lon_r", "bin_30m"])
            .copy())
    df = df.drop(columns=["lat_r", "lon_r", "bin_30m"])
    stages.append(("Dilution (30-min bins)", len(df), before - len(df), "burst duplicates dropped"))

    months_per_year = (df.groupby(["user_id", "year"])["month"]
                         .nunique().reset_index(name="months_in_year"))
    max_months = (months_per_year.groupby("user_id")["months_in_year"]
                                 .max().rename("max_months_in_any_year"))
    df = df.merge(max_months, on="user_id", how="left")
    return df, stages


df, stages = run_pipeline()

st.header("1. Pipeline stages")

stage_rows = []
for name, after, dropped, note in stages:
    pct = f"-{dropped / (after + dropped) * 100:.1f}%" if dropped > 0 else ""
    stage_rows.append({"Stage": name, "Rows after": after, "Dropped": dropped, "%": pct, "Note": note})
st.dataframe(pd.DataFrame(stage_rows), hide_index=True, width="stretch")

# ════════════════════════════════════════════════════════════════════════════════
# 2. TOURIST vs. LOCAL
# ════════════════════════════════════════════════════════════════════════════════

st.divider()
st.header("2. Tourist vs. Local classification")
st.markdown(
    "A user is classified as **local** if in at least one year their photos span "
    "**N or more distinct months**. A tourist typically visits for a week or two — "
    "all within the same month."
)

threshold = st.slider(
    "Minimum months active in a single year to be called **Local**",
    min_value=2, max_value=12, value=2, step=1,
    key="tourist_threshold"
)
df["is_tourist"] = df["max_months_in_any_year"] < threshold

user_summary = (df.groupby("user_id")
                  .agg(max_months_in_any_year=("max_months_in_any_year", "first"),
                       is_tourist=("is_tourist", "first"),
                       n_photos=("photo_id", "count"))
                  .reset_index())

n_tourist = int(user_summary["is_tourist"].sum())
n_local   = int((~user_summary["is_tourist"]).sum())
total     = n_tourist + n_local

col1, col2, col3 = st.columns(3)
col1.metric("Total unique users", f"{total:,}")
col2.metric("Tourists", f"{n_tourist:,}", f"{n_tourist/total*100:.1f}%")
col3.metric("Locals",   f"{n_local:,}",   f"{n_local/total*100:.1f}%")

st.subheader("Max distinct months active in a single year — per user")
st.markdown("The **dashed line** marks the current threshold.")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ax = axes[0]
bins = np.arange(0.5, 13.5, 1)
counts, _, patches = ax.hist(user_summary["max_months_in_any_year"], bins=bins,
                              color="#cccccc", edgecolor="white", linewidth=0.5)
for patch, m in zip(patches, range(1, 13)):
    patch.set_facecolor("#2196F3" if m < threshold else "#4CAF50")
ax.axvline(threshold - 0.5, color="black", linestyle="--", linewidth=1.4,
           label=f"threshold = {threshold}")
ax.set_xlabel("Max distinct months active in any single year")
ax.set_ylabel("Number of users")
ax.set_title("Users by monthly spread (best year)")
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax.set_xticks(range(1, 13))
ax.legend()
for patch, count in zip(patches, counts):
    if count > 0:
        ax.text(patch.get_x() + patch.get_width() / 2, count + total * 0.002,
                f"{int(count):,}", ha="center", va="bottom", fontsize=7)

ax2 = axes[1]
ax2.pie([n_tourist, n_local],
        labels=[f"Tourist\n{n_tourist:,}", f"Local\n{n_local:,}"],
        autopct="%1.1f%%", colors=["#2196F3", "#4CAF50"], startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.5))
ax2.set_title(f"Tourist / Local split  (threshold = {threshold} month{'s' if threshold > 1 else ''})")
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.info(
    f"At threshold **{threshold}**: a user must have photos in at least {threshold} different "
    "months within a single year to be called local. "
    "Look for a natural valley in the histogram to pick the right cut-off."
)

st.divider()
if st.button("Save cleaned CSV with current tourist/local labels"):
    out = df.drop(columns=["max_months_in_any_year"])
    out.to_csv(OUT_PATH, index=False)
    st.success(f"Saved {len(out):,} rows to `{OUT_PATH}`")

# ════════════════════════════════════════════════════════════════════════════════
# 3. POI RADIUS EXPLORER
# ════════════════════════════════════════════════════════════════════════════════

st.divider()
st.header("3. POI Radius Explorer")
st.markdown("""
Each visit is matched to its **nearest OpenStreetMap point of interest (POI)**.
The distance is computed once and cached — the slider below filters instantly.

**POI categories downloaded from OSM:**
| Category | What's included |
|---|---|
| **Tourism** | monuments, museums, viewpoints, hotels |
| **Food** | restaurants, cafes, bars, pubs |
| **Local services** | schools, hospitals, supermarkets, banks |
| **Leisure** | parks, gardens, playgrounds, stadiums |

Use this section to pick a sensible capture radius **before** running clustering.
A visit 80 m from a museum likely isn't "at" that museum — tighter is more honest.
""")


@st.cache_data(show_spinner="Downloading Vienna POIs from OSM...")
def get_pois():
    categories = {
        "tourism": {"tourism": True},
        "food":    {"amenity": ["restaurant", "cafe", "bar", "pub"]},
        "local":   {"amenity": ["school", "hospital", "supermarket", "bank"]},
        "leisure": {"leisure": ["park", "garden", "playground", "stadium"]},
    }
    parts = []
    for cat, tags in categories.items():
        t0 = time.time()
        try:
            gdf = ox.features_from_place("Vienna, Austria", tags=tags)
            gdf = gdf[["geometry"]].copy()
            gdf = gdf.to_crs("EPSG:32633")
            gdf["geometry"] = gdf.geometry.centroid
            gdf = gdf.to_crs("EPSG:4326")
            gdf["poi_category"] = cat
            parts.append(gdf[["geometry", "poi_category"]])
            print(f"  [{time.strftime('%H:%M:%S')}] [POI] {cat}: {len(gdf):,} features in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"  [{time.strftime('%H:%M:%S')}] [POI] {cat}: FAILED ({e})")
    pois = pd.concat(parts, ignore_index=True)
    print(f"  [{time.strftime('%H:%M:%S')}] [POI] total: {len(pois):,} POIs across {len(parts)} categories")
    return gpd.GeoDataFrame(pois, geometry="geometry", crs="EPSG:4326")


@st.cache_data(show_spinner="Computing nearest POI for each visit (one-time, ~30 s)...")
def compute_poi_distances(_df, _pois):
    t0 = time.time()
    gdf_photos = gpd.GeoDataFrame(
        _df[["photo_id"]].copy(),
        geometry=gpd.points_from_xy(_df["long"], _df["lat"]),
        crs="EPSG:4326"
    ).to_crs("EPSG:32633")

    gdf_pois = _pois.to_crs("EPSG:32633")

    result = gpd.sjoin_nearest(
        gdf_photos,
        gdf_pois[["geometry", "poi_category"]],
        how="left",
        distance_col="nearest_poi_dist_m"
    )
    result = result.drop_duplicates(subset=["photo_id"])
    print(f"  [{time.strftime('%H:%M:%S')}] [POI distances] {len(result):,} visits matched in {time.time() - t0:.1f}s")
    return result[["photo_id", "nearest_poi_dist_m", "poi_category"]].reset_index(drop=True)


pois     = get_pois()
poi_dist = compute_poi_distances(df, pois)

st.caption(f"POI dataset: {len(pois):,} points of interest downloaded from OSM across 4 categories")

poi_radius = st.slider(
    "Capture radius (metres) — how close must a visit be to a POI to count?",
    min_value=10, max_value=100, value=40, step=10,
    key="poi_radius"
)

within      = poi_dist[poi_dist["nearest_poi_dist_m"] <= poi_radius]
pct_covered = len(within) / len(poi_dist) * 100

c1, c2, c3 = st.columns(3)
c1.metric("Visits within radius",        f"{len(within):,}", f"{pct_covered:.1f}% of all visits")
c2.metric("Median dist to nearest POI",  f"{poi_dist['nearest_poi_dist_m'].median():.0f} m")
c3.metric("Mean dist to nearest POI",    f"{poi_dist['nearest_poi_dist_m'].mean():.0f} m")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ax = axes[0]
dist_cap = poi_dist["nearest_poi_dist_m"].clip(upper=300)
ax.hist(dist_cap, bins=60, color="#90CAF9", edgecolor="white", linewidth=0.3)
ax.axvline(poi_radius, color="#E53935", linewidth=2, linestyle="--",
           label=f"radius = {poi_radius} m  ({pct_covered:.1f}% covered)")
ax.set_xlabel("Distance to nearest POI (m, capped at 300)")
ax.set_ylabel("Number of visits")
ax.set_title("Distribution of nearest-POI distances")
ax.legend()

ax2 = axes[1]
if len(within) > 0:
    cat_counts = within["poi_category"].value_counts()
    cat_colors = {"tourism": "#E53935", "food": "#FB8C00",
                  "local":   "#43A047", "leisure": "#1E88E5"}
    bars = ax2.barh(cat_counts.index, cat_counts.values,
                    color=[cat_colors.get(c, "#999") for c in cat_counts.index])
    for bar, val in zip(bars, cat_counts.values):
        pct = val / len(within) * 100
        ax2.text(bar.get_width() + len(within) * 0.003,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:,}  ({pct:.1f}%)", va="center", fontsize=9)
    ax2.set_xlabel("Visits within radius")
    ax2.set_title(f"POI category breakdown at {poi_radius} m")
    ax2.set_xlim(right=ax2.get_xlim()[1] * 1.3)
else:
    ax2.text(0.5, 0.5, "No visits within radius", ha="center", va="center",
             transform=ax2.transAxes, fontsize=14, color="gray")

plt.tight_layout()
st.pyplot(fig)
plt.close()

st.info(
    f"At **{poi_radius} m**: {pct_covered:.1f}% of visits are near a POI. "
    "The histogram shows a natural elbow — set the radius just past it. "
    "Too small = most visits unclassified; too large = noise."
)

# ════════════════════════════════════════════════════════════════════════════════
# 4. SPATIAL CLUSTERING
# ════════════════════════════════════════════════════════════════════════════════

st.divider()
st.header("4. Spatial Clustering")
st.markdown(f"""
Tourist behaviour is clustered at the **tourist-day** level — each calendar day
of visits for each tourist is one data point. A tourist visiting Vienna for 4 days
contributes 4 independent rows. This lets us discover *day types* rather than
averaging out behavioural variation across a whole trip.

Features are drawn from three groups, all standardised (z-score) before clustering:

| Group | Features |
|---|---|
| **Mobility** | locations visited · active hours · radius of gyration · path linearity · median step distance · location entropy · % centre locations |
| **POI affinity** | % of locations near each POI category |

POI features use the **{poi_radius} m radius** selected above.
Tourist-days with fewer than 3 visits are excluded (too sparse to characterise movement).

> **Path linearity** measures whether a day's trajectory is a straight-line corridor
> (score ≈ 1) or a loop / area-based exploration (score ≈ 0). Computed as displacement
> from first to last visit divided by total path length.
>
> **Location entropy** measures how spread visits are across ~100 m grid cells.
> High entropy = many distinct spots; low entropy = concentrated in one area.
""")


@st.cache_data(show_spinner="Building per-tourist-day feature vectors...")
def compute_day_features(_df, _poi_dist, poi_radius):
    t0 = time.time()
    df = _df.copy()
    df = df.merge(_poi_dist.rename(columns={"poi_category": "poi_cat_raw"}),
                  on="photo_id", how="left")
    df["poi_cat_in_radius"] = df["poi_cat_raw"].where(
        df["nearest_poi_dist_m"] <= poi_radius
    )
    df["date"]  = df["datetime"].dt.date
    df["lat_r"] = df["lat"].round(3)
    df["lon_r"] = df["long"].round(3)

    records = []
    for (uid, date), grp in df.groupby(["user_id", "date"]):
        grp  = grp.sort_values("datetime")
        lats = grp["lat"].values
        lons = grp["long"].values
        n    = len(grp)

        lat_c = float(lats.mean())
        lon_c = float(lons.mean())

        # Radius of gyration (km)
        dlat = (lats - lat_c) * 111.0
        dlon = (lons - lon_c) * 111.0 * np.cos(np.radians(lat_c))
        rog  = float(np.sqrt((dlat**2 + dlon**2).mean()))

        # Active time span (hours first → last visit)
        time_span = (
            (grp["datetime"].iloc[-1] - grp["datetime"].iloc[0]).total_seconds() / 3600.0
            if n > 1 else 0.0
        )

        # Path linearity & median step distance
        if n > 1:
            displacement = float(_haversine_km(lats[0], lons[0], lats[-1], lons[-1]))
            steps        = _haversine_km(lats[:-1], lons[:-1], lats[1:], lons[1:])
            total_path   = float(steps.sum())
            path_lin     = displacement / total_path if total_path > 0 else 1.0
            median_step  = float(np.median(steps))
        else:
            path_lin    = 1.0
            median_step = 0.0

        # Location entropy over ~100 m grid cells
        cell_labels = [f"{lr},{lo}" for lr, lo in zip(grp["lat_r"].values, grp["lon_r"].values)]
        loc_entropy = _entropy(cell_labels)

        # % of visits within 2 km of Innere Stadt
        dists_c    = _haversine_km(
            lats, lons,
            np.full(n, INNERE_STADT[0]), np.full(n, INNERE_STADT[1])
        )
        pct_center = float((dists_c <= 2.0).mean())

        # POI affinity
        cats      = grp["poi_cat_in_radius"].dropna()
        total_poi = len(cats)
        vc        = cats.value_counts()
        def pct(c): return float(vc.get(c, 0)) / max(total_poi, 1)

        records.append({
            "user_id":               uid,
            "date":                  date,
            "centroid_lat":          lat_c,
            "centroid_lon":          lon_c,
            "n_locations":              n,
            "time_span_hours":       time_span,
            "radius_of_gyration_km": rog,
            "path_linearity":        path_lin,
            "median_step_km":        median_step,
            "location_entropy":      loc_entropy,
            "pct_center_locations":     pct_center,
            "pct_tourism":           pct("tourism"),
            "pct_food":              pct("food"),
            "pct_local":             pct("local"),
            "pct_leisure":           pct("leisure"),
        })

    elapsed = time.time() - t0
    print(f"  [{time.strftime('%H:%M:%S')}] [compute_day_features] {len(records):,} tourist-days in {elapsed:.1f}s")
    return pd.DataFrame(records)


day_feats = compute_day_features(df, poi_dist, poi_radius)

# Merge tourist label (changes with threshold slider — not cached)
is_tourist_map = df.groupby("user_id")["is_tourist"].first().reset_index()
day_feats = day_feats.merge(is_tourist_map, on="user_id", how="left")

n_excluded_days = int((~day_feats["is_tourist"]).sum())
day_feats = day_feats[day_feats["is_tourist"]].copy().reset_index(drop=True)
day_feats = day_feats[day_feats["n_locations"] >= 3].copy().reset_index(drop=True)
st.caption(
    f"Clustering {len(day_feats):,} tourist-days from "
    f"{day_feats['user_id'].nunique():,} tourists "
    f"({n_excluded_days:,} local days excluded). Days with < 3 visits removed."
)

WINSOR_COLS = ["n_locations", "time_span_hours"]
X = day_feats[FEATURE_COLS].fillna(0).copy()
for col in WINSOR_COLS:
    cap = X[col].quantile(0.99)
    X[col] = X[col].clip(upper=cap)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

p99_visits = day_feats["n_locations"].quantile(0.99)
p99_span   = day_feats["time_span_hours"].quantile(0.99)
n_capped   = int((day_feats["n_locations"] > p99_visits).sum())
st.caption(
    f"Feature matrix: {X_scaled.shape[0]:,} tourist-days × {X_scaled.shape[1]} features. "
    f"`n_locations` and `time_span_hours` winsorised at 99th percentile "
    f"({p99_visits:.0f} visits, {p99_span:.1f} h) — "
    f"{n_capped} extreme days remain but no longer dominate a cluster."
)

X_hash = (poi_radius, X_scaled.shape, round(float(X_scaled.sum()), 2))

# ── Why UMAP + HDBSCAN ────────────────────────────────────────────────────────

st.subheader("Clustering approach — UMAP → HDBSCAN")
st.markdown("""
**Why not cluster directly on the 11 features?**

In high-dimensional spaces Euclidean distance loses meaning — a phenomenon
called the *curse of dimensionality*. As the number of features grows, every
pair of points looks roughly equally far apart, undermining any distance-based
algorithm including KMeans and DBSCAN.

**Step 1 — UMAP** compresses the 11 standardised features into 3 dimensions
while preserving *local structure*: days that behave similarly stay close
together in the embedding. The 2-D version below is used for visualisation only.

**Step 2 — HDBSCAN** clusters the 3-D UMAP embedding.

| | KMeans | DBSCAN | **HDBSCAN** |
|---|---|---|---|
| Needs K specified upfront | yes | no | no |
| Handles uneven density | no | no | **yes** |
| Labels outliers as noise | no | yes | **yes** |
| Key parameter | K | eps (hard to calibrate) | min cluster size ✓ |

Tourist-days that don't fit any cluster receive label **−1 (noise)** —
genuinely atypical days worth inspecting. KMeans is kept at the bottom as a
comparison baseline.
""")

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
umap_neighbors = col_ctrl1.slider(
    "n_neighbors — local (low) vs global (high) structure",
    min_value=5, max_value=100, value=25, step=5, key="umap_neighbors",
    help="Raise this if most days collapse into one large cluster — "
         "it widens UMAP's view and separates broad behavioural groups.",
)
min_cs = col_ctrl2.slider(
    "min_cluster_size — smallest group that counts as a cluster",
    min_value=50, max_value=500, value=150, step=50, key="hdb_min_cs",
)
epsilon = col_ctrl3.slider(
    "epsilon — merge clusters closer than ε",
    min_value=0.0, max_value=2.0, value=0.7, step=0.1, key="hdb_epsilon",
    help="Raise to reduce cluster count by merging nearby groups.",
)


@st.cache_data(show_spinner="Running UMAP dimensionality reduction...")
def compute_umap(X_hash, _X_scaled, n_neighbors):
    import umap as umap_lib
    reducer = umap_lib.UMAP(
        n_components=3, random_state=42, n_jobs=1,
        n_neighbors=n_neighbors, min_dist=0.0,
    )
    return reducer.fit_transform(_X_scaled)


embedding = compute_umap(X_hash, X_scaled, umap_neighbors)


@st.cache_data(show_spinner="Running HDBSCAN...")
def compute_hdbscan(X_hash, _embedding, min_cluster_size, epsilon):
    import hdbscan as hdb_lib
    clusterer = hdb_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method="leaf",
    )
    return clusterer.fit_predict(_embedding).tolist()


hdb_labels           = np.array(compute_hdbscan(X_hash, embedding, min_cs, epsilon))
day_feats["cluster"] = hdb_labels

label_s         = pd.Series(hdb_labels)
unique_clusters = sorted([l for l in label_s.unique() if l >= 0])
found_k         = len(unique_clusters)
n_noise         = int((label_s == -1).sum())
n_in_clusters   = len(hdb_labels) - n_noise

c1, c2, c3 = st.columns(3)
c1.metric("Clusters found",       found_k)
c2.metric("Days in clusters",     f"{n_in_clusters:,}")
c3.metric("Noise / outlier days", f"{n_noise:,}", f"{n_noise / len(hdb_labels) * 100:.1f}%")

# ── UMAP 2-D scatter ──────────────────────────────────────────────────────────

st.subheader("UMAP 2-D projection — coloured by HDBSCAN cluster")
st.caption(
    "Each dot is one tourist-day. Proximity = behavioural similarity (16-dimensional "
    "structure projected to 2-D for visualisation). Grey dots are noise days."
)

cmap = {l: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, l in enumerate(unique_clusters)}
cmap[-1] = "#AAAAAA"

fig, ax = plt.subplots(figsize=(9, 6))
for lbl in [-1] + unique_clusters:
    mask = (hdb_labels == lbl)
    name = "Noise" if lbl == -1 else f"Cluster {lbl + 1}"
    ax.scatter(embedding[mask, 0], embedding[mask, 1],
               c=cmap[lbl], s=14, alpha=0.55, linewidths=0,
               label=f"{name}  ({mask.sum():,})")
ax.set_xlabel("UMAP dim 1")
ax.set_ylabel("UMAP dim 2")
ax.set_title("UMAP 2-D — tourist-day behavioural clusters")
ax.legend(markerscale=2, fontsize=9)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Cluster profiles + auto-descriptions ──────────────────────────────────────

if found_k == 0:
    st.warning("HDBSCAN found no clusters at this min_cluster_size — try lowering it.")
else:
    st.subheader("Cluster profiles")
    st.markdown(
        "Each bar shows how far a cluster's average **deviates from the global mean** "
        "in standard deviations. Positive = above average; negative = below average. "
        "Noise days are excluded."
    )

    global_mean = X.mean()
    global_std  = X.std().replace(0, 1e-8)

    z_means = {}
    for c in unique_clusters:
        mask = day_feats["cluster"] == c
        z_means[f"Cluster {c + 1}"] = (X[mask.values].mean() - global_mean) / global_std
    z_df = pd.DataFrame(z_means, index=FEATURE_COLS)

    n_feat     = len(FEATURE_COLS)
    bar_height = 0.8 / found_k

    fig, ax = plt.subplots(figsize=(13, max(6, n_feat * 0.65 + found_k * 0.3)))
    for i, col in enumerate(z_df.columns):
        offsets = np.arange(n_feat) + (i - found_k / 2 + 0.5) * bar_height
        ax.barh(offsets, z_df[col], height=bar_height * 0.88,
                color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                label=col, alpha=0.85)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(FEATURE_COLS, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Z-score vs. global mean")
    ax.set_title("Feature z-scores per cluster (HDBSCAN)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("What does each cluster represent?")
    st.markdown("Top 3 deviating features drive the auto-label.")

    for c in unique_clusters:
        mask   = day_feats["cluster"] == c
        n_days = int(mask.sum())
        n_usrs = int(day_feats.loc[mask, "user_id"].nunique())
        z      = z_df[f"Cluster {c + 1}"]
        top3   = z.abs().nlargest(3).index
        traits = [FEAT_LABELS[f][0] if z[f] > 0 else FEAT_LABELS[f][1] for f in top3]
        color  = CLUSTER_COLORS[unique_clusters.index(c) % len(CLUSTER_COLORS)]
        st.markdown(
            f"<div style='border-left:5px solid {color}; padding:8px 16px; margin:6px 0; "
            f"background:#f9f9f9; border-radius:4px'>"
            f"<b style='font-size:1.05em'>Cluster {c + 1}</b>"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;{n_days:,} tourist-days"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;{n_usrs:,} distinct tourists<br>"
            f"<span style='color:#444; font-size:0.95em'>"
            f"{' &nbsp;·&nbsp; '.join(traits)}</span></div>",
            unsafe_allow_html=True,
        )

    if n_noise > 0:
        st.markdown(
            f"<div style='border-left:5px solid #AAAAAA; padding:8px 16px; margin:6px 0; "
            f"background:#f9f9f9; border-radius:4px'>"
            f"<b>Noise</b> &nbsp;|&nbsp; {n_noise:,} tourist-days not assigned to any cluster<br>"
            f"<span style='color:#444; font-size:0.95em'>"
            f"Atypical or sparse behavioural patterns — contradictory signals across "
            f"features. Worth inspecting manually.</span></div>",
            unsafe_allow_html=True,
        )

    # ── Tourist meta-types ────────────────────────────────────────────────────

    st.subheader("Tourist meta-types")
    st.markdown("""
A tourist visiting Vienna over multiple days may show *different* day-types across
their trip — perhaps a linear sightseeing day followed by a leisurely neighbourhood
wander. Their **meta-type** is the day-type they exhibit most often — their
behavioural tendency. The distribution below shows how many tourists lean toward
each type.
""")

    assigned = day_feats[day_feats["cluster"] >= 0].copy()
    if len(assigned) > 0:
        meta = (assigned.groupby("user_id")["cluster"]
                .agg(lambda x: x.mode().iloc[0])
                .reset_index(name="meta_cluster"))

        meta_counts = meta["meta_cluster"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 3.5))
        bar_colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in meta_counts.index]
        bars = ax.bar([f"Cluster {c + 1}" for c in meta_counts.index],
                      meta_counts.values, color=bar_colors)
        for bar, val in zip(bars, meta_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(val), ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Number of tourists")
        ax.set_title("Tourist meta-type distribution (modal day-type per tourist)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Per-tourist day-type breakdown
        pivot = (assigned.groupby(["user_id", "cluster"])
                 .size()
                 .unstack(fill_value=0)
                 .rename(columns=lambda c: f"C{c + 1}"))
        pivot["total_days"] = pivot.sum(axis=1)
        pivot["meta_type"]  = (assigned.groupby("user_id")["cluster"]
                                .agg(lambda x: f"C{x.mode().iloc[0] + 1}"))
        pivot = pivot.sort_values("total_days", ascending=False)
        st.caption(
            f"{len(pivot):,} tourists with ≥1 assigned day. "
            "Tourists with counts spread across columns have mixed behaviour."
        )
        st.dataframe(pivot.reset_index(), hide_index=True, width="stretch")

    # ── Cluster map ───────────────────────────────────────────────────────────

    st.subheader("Where does each cluster photograph?")
    st.markdown(
        "Each dot is one **tourist-day centroid** (average location of all visits that day). "
        "Grey = noise. Click cluster names in the legend to toggle layers."
    )

    map_records = [
        {
            "lat":     float(row["centroid_lat"]),
            "lon":     float(row["centroid_lon"]),
            "cluster": "Noise" if int(row["cluster"]) == -1 else f"Cluster {int(row['cluster']) + 1}",
            "color":   cmap.get(int(row["cluster"]), "#AAAAAA"),
        }
        for _, row in day_feats[["centroid_lat", "centroid_lon", "cluster"]].iterrows()
    ]
    all_labels      = ["Noise"] + [f"Cluster {c + 1}" for c in unique_clusters]
    label_color_map = {"Noise": "#AAAAAA"}
    label_color_map.update({
        f"Cluster {c + 1}": CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        for i, c in enumerate(unique_clusters)
    })

    map_html = """
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<div id="map" style="height:520px; border-radius:6px;"></div>
<script>
const DATA   = """ + json.dumps(map_records) + """;
const LABELS = """ + json.dumps(all_labels) + """;
const CMAP   = """ + json.dumps(label_color_map) + """;

var map = L.map('map').setView([48.205, 16.37], 11);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19
}).addTo(map);

var overlays = {};
var layerList = [];
LABELS.forEach(function(lbl) {
    var color = CMAP[lbl] || '#999';
    var layer = L.layerGroup();
    DATA.filter(function(d) { return d.cluster === lbl; }).forEach(function(d) {
        L.circleMarker([d.lat, d.lon], {
            radius: 5, color: color, fillColor: color,
            fillOpacity: 0.65, weight: 0.5
        }).bindTooltip(lbl).addTo(layer);
    });
    overlays['<span style="color:' + color + '; font-size:1.2em">&#9632;</span> ' + lbl] = layer;
    layerList.push(layer);
    layer.addTo(map);
});

L.control.layers(null, overlays, {collapsed: false, position: 'topright'}).addTo(map);

var BtnControl = L.Control.extend({
    options: { position: 'topright' },
    onAdd: function() {
        var div = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
        div.style.cssText = 'background:white; padding:4px 8px; display:flex; gap:6px; box-shadow:0 1px 5px rgba(0,0,0,0.3);';
        ['Select all', 'Clear all'].forEach(function(label) {
            var btn = L.DomUtil.create('button', '', div);
            btn.innerHTML = label;
            btn.style.cssText = 'cursor:pointer; font-size:12px; padding:2px 8px; border:1px solid #ccc; border-radius:3px; background:#f8f8f8;';
            L.DomEvent.on(btn, 'click', L.DomEvent.stopPropagation);
            L.DomEvent.on(btn, 'click', function() {
                layerList.forEach(function(l) {
                    label === 'Select all' ? map.addLayer(l) : map.removeLayer(l);
                });
            });
        });
        return div;
    }
});
new BtnControl().addTo(map);
</script>
"""
    st.iframe(map_html, height=540)

    # ── Summary table ─────────────────────────────────────────────────────────

    st.subheader("Cluster summary table")
    summary_rows = []
    for c in unique_clusters:
        mask = day_feats["cluster"] == c
        grp  = day_feats[mask]
        summary_rows.append({
            "Cluster":           str(c + 1),
            "Tourist-days":      int(mask.sum()),
            "Distinct tourists": int(grp["user_id"].nunique()),
            "% of days":         f"{mask.sum() / len(day_feats) * 100:.1f}%",
            "Avg visits":        f"{grp['n_locations'].mean():.1f}",
            "Avg span (h)":      f"{grp['time_span_hours'].mean():.1f}",
            "Avg radius (km)":   f"{grp['radius_of_gyration_km'].mean():.2f}",
            "Avg linearity":     f"{grp['path_linearity'].mean():.2f}",
        })
    if n_noise > 0:
        mask = day_feats["cluster"] == -1
        grp  = day_feats[mask]
        summary_rows.append({
            "Cluster":           "Noise",
            "Tourist-days":      int(mask.sum()),
            "Distinct tourists": int(grp["user_id"].nunique()),
            "% of days":         f"{mask.sum() / len(day_feats) * 100:.1f}%",
            "Avg visits":        f"{grp['n_locations'].mean():.1f}",
            "Avg span (h)":      f"{grp['time_span_hours'].mean():.1f}",
            "Avg radius (km)":   f"{grp['radius_of_gyration_km'].mean():.2f}",
            "Avg linearity":     f"{grp['path_linearity'].mean():.2f}",
        })
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

# ════════════════════════════════════════════════════════════════════════════════
# 5. KMeans comparison
# ════════════════════════════════════════════════════════════════════════════════

st.divider()
st.header("5. KMeans comparison")
st.markdown("""
KMeans is kept here as a **baseline** to contrast with HDBSCAN. Key things
to look for in the cross-tabulation below:

- **Diagonal pattern** → both algorithms agree; those clusters are robust.
- **Off-diagonal mass** → days where the algorithms disagree — often the most
  analytically interesting boundary-case behaviour.
- KMeans forces *every* day into a cluster (no noise class), so atypical days
  inflate whichever centroid they happen to be nearest to.
""")


@st.cache_data(show_spinner="Running KMeans for K = 2..7...")
def compute_elbow(X_hash, _X_scaled):
    ks, inertias, silhouettes = [], [], []
    for k in range(2, 8):
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(_X_scaled)
        ks.append(k)
        inertias.append(km.inertia_)
        silhouettes.append(
            silhouette_score(_X_scaled, labels,
                             sample_size=min(5000, len(_X_scaled)),
                             random_state=42)
        )
    return ks, inertias, silhouettes


ks, inertias, silhouettes = compute_elbow(X_hash, X_scaled)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
ax1.plot(ks, inertias,    "o-", color="#1E88E5")
ax1.set_xlabel("K");  ax1.set_ylabel("Inertia")
ax1.set_title("Elbow curve — look for the bend")
ax1.xaxis.set_major_locator(mticker.MultipleLocator(1))

ax2.plot(ks, silhouettes, "o-", color="#43A047")
ax2.set_xlabel("K");  ax2.set_ylabel("Silhouette score (higher = better)")
ax2.set_title("Silhouette score")
ax2.xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.info(
    "**How to choose K:** find the elbow in the inertia curve (where improvement flattens), "
    "confirmed by a silhouette peak. Silhouette > 0.10 is acceptable for geospatial data."
)

k_choice = st.slider("KMeans K", min_value=2, max_value=7, value=4, step=1, key="km_k")
km_final = KMeans(n_clusters=k_choice, random_state=42, n_init=10)
day_feats["cluster_kmeans"] = km_final.fit_predict(X_scaled)

# ── Cross-tabulation heatmap ──────────────────────────────────────────────────

if found_k > 0:
    st.subheader("HDBSCAN vs KMeans — agreement heatmap")
    st.markdown("""
Each cell shows how many tourist-days were assigned to that **HDBSCAN cluster** *and*
that **KMeans cluster**. A strong diagonal means both algorithms agree. High
off-diagonal cells are the most interesting — days sitting ambiguously between groups.
""")

    cross = pd.crosstab(
        day_feats["cluster"].map(
            lambda x: "HDB Noise" if x == -1 else f"HDB {int(x) + 1}"
        ),
        day_feats["cluster_kmeans"].map(lambda x: f"KM {int(x) + 1}"),
    )

    n_hdb_rows = found_k + (1 if n_noise > 0 else 0)
    fig, ax = plt.subplots(figsize=(max(5, k_choice * 1.3), max(4, n_hdb_rows * 0.9 + 1.5)))
    im  = ax.imshow(cross.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(cross.columns)));  ax.set_xticklabels(cross.columns, fontsize=9)
    ax.set_yticks(range(len(cross.index)));    ax.set_yticklabels(cross.index, fontsize=9)
    ax.set_xlabel("KMeans cluster");           ax.set_ylabel("HDBSCAN cluster")
    ax.set_title("Agreement heatmap — HDBSCAN × KMeans")
    plt.colorbar(im, ax=ax, label="Tourist-days")
    vmax = cross.values.max()
    for i in range(len(cross.index)):
        for j in range(len(cross.columns)):
            val = cross.values[i, j]
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8,
                        color="white" if val > vmax * 0.5 else "black")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Save ──────────────────────────────────────────────────────────────────────

st.divider()
if st.button("Save clustered CSV (adds 'cluster' and 'cluster_kmeans' columns)"):
    df_save = df.copy()
    df_save["date"] = df_save["datetime"].dt.date
    merged = df_save.merge(
        day_feats[["user_id", "date", "cluster", "cluster_kmeans"]],
        on=["user_id", "date"], how="left"
    )
    merged.drop(columns=["max_months_in_any_year"], errors="ignore").to_csv(CLUSTER_PATH, index=False)
    st.success(f"Saved {len(merged):,} rows to `{CLUSTER_PATH}`")
