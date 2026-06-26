"""
Flickr Vienna – preprocessing + POI explorer + spatial clustering
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point, MultiPoint
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RAW_PATH     = r"c:\projects\module_4\simulator\simulator_data\Vienna.txt"
OUT_PATH     = r"c:\projects\module_4\simulator\simulator_data\Vienna_clean.csv"
CLUSTER_PATH = r"c:\projects\module_4\simulator\simulator_data\Vienna_clustered.csv"
INNERE_STADT = (48.2093, 16.3728)   # geographic centre of Innere Stadt

FEATURE_COLS = [
    "n_visits", "n_locations", "radius_of_gyration_km",
    "centroid_dist_center_km", "convex_hull_area_km2", "max_day_dist_km",
    "evening_ratio", "morning_ratio", "weekend_ratio",
    "pct_tourism", "pct_food", "pct_local", "pct_leisure", "poi_entropy",
]

# (positive label, negative label) — used to auto-describe cluster traits
FEAT_LABELS = {
    "n_visits":                 ("many visit events",        "few visit events"),
    "n_locations":              ("many distinct spots",     "concentrated in one area"),
    "radius_of_gyration_km":    ("wide geographic spread",  "narrow spread"),
    "centroid_dist_center_km":  ("far from city centre",    "city-centre focused"),
    "convex_hull_area_km2":     ("large territory",         "small territory"),
    "max_day_dist_km":          ("long daily explorations", "short daily range"),
    "evening_ratio":            ("evening-active",          "avoids evenings"),
    "morning_ratio":            ("morning-active",          "avoids mornings"),
    "weekend_ratio":            ("weekend-active",          "weekday-active"),
    "pct_tourism":              ("near tourist POIs",       "avoids tourist spots"),
    "pct_food":                 ("food & drink oriented",   "avoids food venues"),
    "pct_local":                ("local services area",     "avoids local services"),
    "pct_leisure":              ("leisure & parks",         "avoids leisure areas"),
    "poi_entropy":              ("diverse POI interests",   "specialised POI focus"),
}

CLUSTER_COLORS = ["#E53935", "#1E88E5", "#43A047", "#FB8C00",
                  "#8E24AA", "#00ACC1", "#F4511E"]

st.set_page_config(page_title="Flickr Vienna", layout="wide")
st.title("Flickr Vienna – Preprocessing & Spatial Analysis")
st.markdown(
    "A 5-stage cleaning pipeline, tourist/local classification, "
    "POI coverage explorer, and unsupervised user clustering based on spatial behaviour."
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
Each photo is matched to its **nearest OpenStreetMap point of interest (POI)**.
The distance is computed once and cached — the slider below filters instantly.

**POI categories downloaded from OSM:**
| Category | What's included |
|---|---|
| **Tourism** | monuments, museums, viewpoints, hotels |
| **Food** | restaurants, cafes, bars, pubs |
| **Local services** | schools, hospitals, supermarkets, banks |
| **Leisure** | parks, gardens, playgrounds, stadiums |

Use this section to pick a sensible capture radius **before** running clustering.
A photo 80 m from a museum likely isn't "at" that museum — tighter is more honest.
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
            gdf["geometry"] = gdf.geometry.centroid   # centroid in metric CRS (correct)
            gdf = gdf.to_crs("EPSG:4326")
            gdf["poi_category"] = cat
            parts.append(gdf[["geometry", "poi_category"]])
            print(f"  [POI] {cat}: {len(gdf):,} features in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"  [POI] {cat}: FAILED ({e})")
    pois = pd.concat(parts, ignore_index=True)
    print(f"  [POI] total: {len(pois):,} POIs across {len(parts)} categories")
    return gpd.GeoDataFrame(pois, geometry="geometry", crs="EPSG:4326")


@st.cache_data(show_spinner="Computing nearest POI for each photo (one-time, ~30 s)...")
def compute_poi_distances(_df, _pois):
    """Returns a DataFrame with photo_id, nearest_poi_dist_m, poi_category."""
    t0 = time.time()
    gdf_photos = gpd.GeoDataFrame(
        _df[["photo_id"]].copy(),
        geometry=gpd.points_from_xy(_df["long"], _df["lat"]),
        crs="EPSG:4326"
    ).to_crs("EPSG:32633")           # UTM 33N — metres, suitable for Austria

    gdf_pois = _pois.to_crs("EPSG:32633")

    result = gpd.sjoin_nearest(
        gdf_photos,
        gdf_pois[["geometry", "poi_category"]],
        how="left",
        distance_col="nearest_poi_dist_m"
    )
    result = result.drop_duplicates(subset=["photo_id"])
    print(f"  [POI distances] {len(result):,} photos matched in {time.time() - t0:.1f}s")
    return result[["photo_id", "nearest_poi_dist_m", "poi_category"]].reset_index(drop=True)


pois    = get_pois()
poi_dist = compute_poi_distances(df, pois)

st.caption(f"POI dataset: {len(pois):,} points of interest downloaded from OSM across 4 categories")

poi_radius = st.slider(
    "Capture radius (metres) — how close must a photo be to a POI to count?",
    min_value=10, max_value=100, value=50, step=10,
    key="poi_radius"
)

within      = poi_dist[poi_dist["nearest_poi_dist_m"] <= poi_radius]
pct_covered = len(within) / len(poi_dist) * 100

c1, c2, c3 = st.columns(3)
c1.metric("Photos within radius", f"{len(within):,}", f"{pct_covered:.1f}% of all photos")
c2.metric("Median dist to nearest POI", f"{poi_dist['nearest_poi_dist_m'].median():.0f} m")
c3.metric("Mean dist to nearest POI",   f"{poi_dist['nearest_poi_dist_m'].mean():.0f} m")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Left: histogram of nearest-POI distances
ax = axes[0]
dist_cap = poi_dist["nearest_poi_dist_m"].clip(upper=300)
ax.hist(dist_cap, bins=60, color="#90CAF9", edgecolor="white", linewidth=0.3)
ax.axvline(poi_radius, color="#E53935", linewidth=2, linestyle="--",
           label=f"radius = {poi_radius} m  ({pct_covered:.1f}% covered)")
ax.set_xlabel("Distance to nearest POI (m, capped at 300)")
ax.set_ylabel("Number of photos")
ax.set_title("Distribution of nearest-POI distances")
ax.legend()

# Right: category breakdown at chosen radius
ax2 = axes[1]
if len(within) > 0:
    cat_counts  = within["poi_category"].value_counts()
    cat_colors  = {"tourism": "#E53935", "food": "#FB8C00",
                   "local":   "#43A047", "leisure": "#1E88E5"}
    bars = ax2.barh(cat_counts.index, cat_counts.values,
                    color=[cat_colors.get(c, "#999") for c in cat_counts.index])
    for bar, val in zip(bars, cat_counts.values):
        pct = val / len(within) * 100
        ax2.text(bar.get_width() + len(within) * 0.003,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:,}  ({pct:.1f}%)", va="center", fontsize=9)
    ax2.set_xlabel("Photos within radius")
    ax2.set_title(f"POI category breakdown at {poi_radius} m")
    ax2.set_xlim(right=ax2.get_xlim()[1] * 1.3)
else:
    ax2.text(0.5, 0.5, "No photos within radius", ha="center", va="center",
             transform=ax2.transAxes, fontsize=14, color="gray")

plt.tight_layout()
st.pyplot(fig)
plt.close()

st.info(
    f"At **{poi_radius} m**: {pct_covered:.1f}% of photos are near a POI. "
    "The histogram shows a natural elbow — set the radius just past it. "
    "Too small = most photos unclassified; too large = noise "
    "(a photo 90 m away may have nothing to do with that POI)."
)

# ════════════════════════════════════════════════════════════════════════════════
# 4. SPATIAL CLUSTERING
# ════════════════════════════════════════════════════════════════════════════════

st.divider()
st.header("4. Spatial Clustering")
st.markdown(f"""
Users are clustered by **how they move through Vienna**, not just where.
Features come from three groups — all standardised (z-score) before clustering.

| Group | Features |
|---|---|
| **Mobility** | visit events · distinct locations · radius of gyration · convex hull area · max daily exploration · centroid distance from Innere Stadt |
| **Temporal** | morning / evening / weekend activity ratios |
| **POI affinity** | % of photos near each category · POI entropy (diversity of interests) |

POI features use the **{poi_radius} m radius** selected above.
Users with fewer than 3 visit events are excluded (too sparse to characterise).

> **visit events vs. distinct locations:** preprocessing already removed burst shots
> (one kept per user × location × 30-min window), so *visit events* counts how many
> separate location-time slots a user has — not raw photos.
> *Distinct locations* further collapses time: the same spot visited on Monday and
> Friday counts as 1 location but 2 visit events.
> A local who revisits the same café daily scores high on visits but moderate on locations;
> an explorer tourist scores similarly on both.
""")


@st.cache_data(show_spinner="Building per-user feature vectors (may take ~30 s)...")
def compute_user_features(_df, _poi_dist, poi_radius):
    df = _df.copy()
    df = df.merge(_poi_dist.rename(columns={"poi_category": "poi_cat_raw"}),
                  on="photo_id", how="left")
    df["poi_cat_in_radius"] = df["poi_cat_raw"].where(
        df["nearest_poi_dist_m"] <= poi_radius
    )
    df["date"]  = df["datetime"].dt.date
    df["hour"]  = df["datetime"].dt.hour
    df["dow"]   = df["datetime"].dt.dayofweek
    df["lat_r"] = df["lat"].round(4)
    df["lon_r"] = df["long"].round(4)

    records = []
    for uid, grp in df.groupby("user_id"):
        lats  = grp["lat"].values
        lons  = grp["long"].values
        lat_c = float(lats.mean())
        lon_c = float(lons.mean())
        n     = len(grp)

        # Radius of gyration (km)
        dlat = (lats - lat_c) * 111.0
        dlon = (lons - lon_c) * 111.0 * np.cos(np.radians(lat_c))
        rog  = float(np.sqrt((dlat**2 + dlon**2).mean()))

        # Convex hull area (km²)
        if n >= 3:
            pts       = MultiPoint(list(zip(lons, lats)))
            hull_area = float(pts.convex_hull.area * 111.0**2 * np.cos(np.radians(lat_c)))
        else:
            hull_area = 0.0

        # Centroid → Innere Stadt (haversine, km)
        lat1, lon1 = np.radians(lat_c),         np.radians(lon_c)
        lat2, lon2 = np.radians(INNERE_STADT[0]), np.radians(INNERE_STADT[1])
        a = (np.sin((lat2 - lat1) / 2) ** 2
             + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
        dist_center = float(6371.0 * 2 * np.arcsin(np.sqrt(a)))

        # Max daily bbox diagonal (km)
        day_g    = grp.groupby("date")
        lat_span = (day_g["lat"].max()  - day_g["lat"].min())  * 111.0
        lon_span = (day_g["long"].max() - day_g["long"].min()) * 111.0 * np.cos(np.radians(lat_c))
        max_day  = float(np.sqrt(lat_span**2 + lon_span**2).max())

        # Temporal ratios
        hours = grp["hour"].values
        dow   = grp["dow"].values

        # POI affinity
        cats      = grp["poi_cat_in_radius"].dropna()
        total_poi = len(cats)
        vc        = cats.value_counts()

        def pct(c):
            return float(vc.get(c, 0)) / max(total_poi, 1)

        poi_ent = 0.0
        if total_poi > 1 and len(vc) > 1:
            p       = vc / vc.sum()
            poi_ent = float(-(p * np.log2(p + 1e-10)).sum())

        records.append({
            "user_id":                 uid,
            "centroid_lat":            lat_c,
            "centroid_lon":            lon_c,
            "n_visits":                n,
            "n_locations":             grp[["lat_r", "lon_r"]].drop_duplicates().shape[0],
            "radius_of_gyration_km":   rog,
            "centroid_dist_center_km": dist_center,
            "convex_hull_area_km2":    hull_area,
            "max_day_dist_km":         max_day,
            "evening_ratio":           float((hours >= 18).mean()),
            "morning_ratio":           float((hours < 12).mean()),
            "weekend_ratio":           float((dow >= 5).mean()),
            "pct_tourism":             pct("tourism"),
            "pct_food":                pct("food"),
            "pct_local":               pct("local"),
            "pct_leisure":             pct("leisure"),
            "poi_entropy":             poi_ent,
        })

    return pd.DataFrame(records)


user_feats = compute_user_features(df, poi_dist, poi_radius)
user_feats = user_feats[user_feats["n_visits"] >= 3].copy().reset_index(drop=True)

# Merge current tourist label (changes with threshold slider — not cached)
is_tourist_map = df.groupby("user_id")["is_tourist"].first().reset_index()
user_feats = user_feats.merge(is_tourist_map, on="user_id", how="left")

exclude_locals = st.checkbox("Cluster tourists only (exclude locals)", value=False)
if exclude_locals:
    n_excluded = int((~user_feats["is_tourist"]).sum())
    user_feats = user_feats[user_feats["is_tourist"]].copy().reset_index(drop=True)
    st.caption(f"Excluded {n_excluded:,} local users — clustering {len(user_feats):,} tourists.")

# Winsorize n_visits and n_locations at 99th percentile so a handful of
# extreme power-users don't pull an entire cluster to themselves.
WINSOR_COLS = ["n_visits", "n_locations"]
X = user_feats[FEATURE_COLS].fillna(0).copy()
for col in WINSOR_COLS:
    cap = X[col].quantile(0.99)
    X[col] = X[col].clip(upper=cap)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

p99_visits = user_feats["n_visits"].quantile(0.99)
p99_locs   = user_feats["n_locations"].quantile(0.99)
n_capped   = int((user_feats["n_visits"] > p99_visits).sum())
st.caption(
    f"Feature matrix: {X_scaled.shape[0]:,} users x {X_scaled.shape[1]} features. "
    f"`n_visits` and `n_locations` winsorised at the 99th percentile "
    f"({p99_visits:.0f} and {p99_locs:.0f} respectively) — "
    f"{n_capped} extreme users remain in the data but no longer dominate a cluster."
)

# ── Elbow + silhouette ────────────────────────────────────────────────────────

# Cache key: a tuple that uniquely identifies this scaled dataset
X_hash = (poi_radius, X_scaled.shape, round(float(X_scaled.sum()), 2))


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

ax1.plot(ks, inertias, "o-", color="#1E88E5")
ax1.set_xlabel("K (number of clusters)")
ax1.set_ylabel("Inertia")
ax1.set_title("Elbow curve — look for the bend")
ax1.xaxis.set_major_locator(mticker.MultipleLocator(1))

ax2.plot(ks, silhouettes, "o-", color="#43A047")
ax2.set_xlabel("K")
ax2.set_ylabel("Silhouette score (higher = better)")
ax2.set_title("Silhouette score")
ax2.xaxis.set_major_locator(mticker.MultipleLocator(1))

plt.tight_layout()
st.pyplot(fig)
plt.close()

st.info(
    "**How to choose K:** find the elbow in the inertia curve (where improvement flattens), "
    "confirmed by a silhouette peak. Silhouette > 0.10 is acceptable for geospatial data."
)

# ── K selector ────────────────────────────────────────────────────────────────

k_choice = st.slider("Number of clusters (K)", min_value=2, max_value=7, value=4, step=1)

km_final = KMeans(n_clusters=k_choice, random_state=42, n_init=10)
user_feats["cluster"] = km_final.fit_predict(X_scaled)

# ── Cluster profile chart (z-score heatmap) ───────────────────────────────────

st.subheader("Cluster profiles")
st.markdown(
    "Each bar shows how far a cluster's average **deviates from the global mean** "
    "in standard deviations. Positive = above average; negative = below average."
)

global_mean = X.mean()
global_std  = X.std().replace(0, 1e-8)

z_means = {}
for c in range(k_choice):
    mask = user_feats["cluster"] == c
    z_means[f"Cluster {c+1}"] = (X[mask.values].mean() - global_mean) / global_std

z_df = pd.DataFrame(z_means, index=FEATURE_COLS)

n_feat     = len(FEATURE_COLS)
bar_height = 0.8 / k_choice

fig, ax = plt.subplots(figsize=(13, max(5, n_feat * 0.55)))
for i, col in enumerate(z_df.columns):
    offsets = np.arange(n_feat) + (i - k_choice / 2 + 0.5) * bar_height
    ax.barh(offsets, z_df[col], height=bar_height * 0.88,
            color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
            label=col, alpha=0.85)

ax.set_yticks(range(n_feat))
ax.set_yticklabels(FEATURE_COLS, fontsize=9)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Z-score vs. global mean")
ax.set_title("Feature z-scores per cluster")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Auto-generated cluster descriptions ───────────────────────────────────────

st.subheader("What does each cluster represent?")
st.markdown(
    "The **three features that deviate most** from the global average "
    "are used to auto-label each cluster in plain English."
)

for c in range(k_choice):
    mask    = user_feats["cluster"] == c
    n_users = int(mask.sum())
    t_pct   = float(user_feats.loc[mask, "is_tourist"].mean() * 100)
    z       = z_df[f"Cluster {c+1}"]
    top3    = z.abs().nlargest(3).index
    traits  = [FEAT_LABELS[f][0] if z[f] > 0 else FEAT_LABELS[f][1] for f in top3]
    color   = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]

    st.markdown(
        f"<div style='border-left:5px solid {color}; padding:8px 16px; margin:6px 0; "
        f"background:#f9f9f9; border-radius:4px'>"
        f"<b style='font-size:1.05em'>Cluster {c+1}</b>"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;{n_users:,} users"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;{t_pct:.0f}% tourist<br>"
        f"<span style='color:#444; font-size:0.95em'>"
        f"{' &nbsp;·&nbsp; '.join(traits)}</span></div>",
        unsafe_allow_html=True,
    )

# ── Cluster map ───────────────────────────────────────────────────────────────

st.subheader("Where does each cluster photograph?")
st.markdown(
    "Each dot is a **user's photo centroid** — the centre of mass of all their photos. "
    "Click cluster names in the legend to toggle them on/off."
)

map_records = (
    user_feats[["centroid_lat", "centroid_lon", "cluster"]]
    .assign(cluster=user_feats["cluster"] + 1)   # 1-based for display
    .rename(columns={"centroid_lat": "lat", "centroid_lon": "lon"})
    .to_dict(orient="records")
)
display_colors = CLUSTER_COLORS[:k_choice]

map_html = """
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<div id="map" style="height:520px; border-radius:6px;"></div>
<script>
const DATA   = """ + json.dumps(map_records) + """;
const COLORS = """ + json.dumps(display_colors) + """;
const K      = """ + str(k_choice) + """;

var map = L.map('map').setView([48.205, 16.37], 11);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19
}).addTo(map);

var overlays = {};
var layerList = [];
for (var c = 1; c <= K; c++) {
    var color = COLORS[(c - 1) % COLORS.length];
    var layer = L.layerGroup();
    DATA.filter(function(d) { return d.cluster === c; }).forEach(function(d) {
        L.circleMarker([d.lat, d.lon], {
            radius: 5,
            color: color,
            fillColor: color,
            fillOpacity: 0.65,
            weight: 0.5
        }).bindTooltip('Cluster ' + c).addTo(layer);
    });
    var label = '<span style="color:' + color + '; font-size:1.2em">&#9632;</span> Cluster ' + c;
    overlays[label] = layer;
    layerList.push(layer);
    layer.addTo(map);
}

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

components.html(map_html, height=540)

# ── Summary table ─────────────────────────────────────────────────────────────

st.subheader("Cluster summary table")

summary_rows = []
for c in range(k_choice):
    mask = user_feats["cluster"] == c
    grp  = user_feats[mask]
    summary_rows.append({
        "Cluster":          c + 1,
        "Users":            int(mask.sum()),
        "% of users":       f"{mask.sum() / len(user_feats) * 100:.1f}%",
        "Avg visits":       f"{grp['n_visits'].mean():.1f}",
        "Avg locations":    f"{grp['n_locations'].mean():.1f}",
        "Avg radius (km)":  f"{grp['radius_of_gyration_km'].mean():.2f}",
        "% tourists":       f"{grp['is_tourist'].mean() * 100:.1f}%",
    })

st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

# ── Save ──────────────────────────────────────────────────────────────────────

st.divider()
if st.button("Save clustered CSV (adds 'cluster' column to Vienna_clean.csv)"):
    merged = df.merge(user_feats[["user_id", "cluster"]], on="user_id", how="left")
    merged.drop(columns=["max_months_in_any_year"], errors="ignore").to_csv(CLUSTER_PATH, index=False)
    st.success(f"Saved {len(merged):,} rows to `{CLUSTER_PATH}`")
