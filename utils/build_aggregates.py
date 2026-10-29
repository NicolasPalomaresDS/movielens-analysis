"""Build the CSV aggregates that feed the Looker Studio dashboard.

Reads the clean parquet files (ratings, movies, tags)
and writes 9 small CSVs.
The dashboard never sees the 32M-row ratings table.

Usage:
    python build_aggregates.py
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# IMAX is a format, not a genre
EXCLUDED_GENRES = {"(no genres listed)", "IMAX"}

# Bayesian prior strength ("virtual" ratings at the global mean)
PRIOR_C = 500

# movies below this are left out of movie_genre.csv
MIN_RATINGS = 20

# genre x decade cells below this are dropped (too noisy)
MIN_CELL_RATINGS = 1000

N_TOP_TAGS = 20

WEEKDAYS = [
    "Mon",
    "Tue",
    "Wed",
    "Thu",
    "Fri",
    "Sat",
    "Sun"
]

# Tier definitions: edges are left-closed [a, b);
# labels line up with the bins.
MOVIE_FILTER_EDGES = [
    0,   5,    10,
    20,  50,   100,
    500, 1000, 10000,
    np.inf
]

MOVIE_FILTER_LABELS = [
    "1-4",     "5-9",         "10-19",
    "20-49",   "50-99",       "100-499",
    "500-999", "1,000-9,999", "10,000+"
]

MOVIE_TAIL_EDGES = [
    1,  2,   5,
    10, 100, 1000,
    10000, np.inf
]

MOVIE_TAIL_LABELS = [
    "1",     "2-4",     "5-9",
    "10-99", "100-999", "1,000-9,999",
    "10,000+"
]

USER_TAIL_EDGES = [
    0,   20,  50,
    100, 500, 1000,
    np.inf
]

USER_TAIL_LABELS = [
    "<20",     "20-49",   "50-99",
    "100-499", "500-999", "1,000+"
]


def tier_index(values, edges):
    return pd.cut(
        values,
        bins=edges,
        labels=False,
        right=False
    )


def main(clean: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    def save(df: pd.DataFrame, name: str) -> None:
        path = out / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8")

        print(
            f"{name + '.csv':<20}{len(df):>9,}",
            f"rows{path.stat().st_size / 1e3:>10.1f} KB"
        )

    # load
    ratings = pd.read_parquet(
        clean / "ratings.parquet",

        columns=[
            "user_id",
            "movie_id",
            "rating",
            "timestamp"
        ]
    )

    movies = pd.read_parquet(clean / "movies.parquet")
    tags = pd.read_parquet(clean / "tags.parquet")

    uid = ratings["user_id"].to_numpy()
    mid = ratings["movie_id"].to_numpy()
    r = ratings["rating"].to_numpy()
    ts = ratings["timestamp"]

    year = ts.dt.year.to_numpy().astype(np.int64)
    weekday = ts.dt.dayofweek.to_numpy().astype(np.int64)
    hour = ts.dt.hour.to_numpy().astype(np.int64)
    n_ratings = len(ratings)
    min_ts, max_ts = ts.min(), ts.max()

    del ratings

    # per-movie / per-user sums
    cnt = np.bincount(mid)

    # float64 weights: exact sums
    tot = np.bincount(mid, weights=r)
    ids = np.flatnonzero(cnt)
    u_cnt = np.bincount(uid)
    u_cnt = u_cnt[u_cnt > 0]
    n_users, n_movies = len(u_cnt), len(ids)
    global_mean = float(tot.sum() / cnt.sum())

    stats = pd.DataFrame({
        "movie_id": ids,
        "count": cnt[ids],
        "total": tot[ids]
    })

    stats = stats.merge(
        movies,
        on="movie_id",
        how="left",
        validate="one_to_one"
    )

    assert (
        stats["title"].notna().all()
    ), "ratings contain movie_ids missing from movies"

    stats["year"] = pd.to_numeric(
        stats["title"]
        .str
        .extract(r"\((\d{4})\)\s*$")[0]
    ).astype("Int16")

    # movie x genre long table
    # (IMAX and "(no genres listed)" excluded)
    ge = stats.assign(
        genre=stats["genres"].str.split("|")
    ).explode("genre")

    ge = ge[~ge["genre"].isin(EXCLUDED_GENRES)]

    # 1. kpis

    kpis = pd.DataFrame([{
        "n_ratings": n_ratings,
        "n_users": n_users,
        "n_movies_rated": n_movies,
        "n_movies_catalog": len(movies),
        "n_tags": len(tags),
        "n_distinct_tags": tags["tag"].nunique(),
        "global_mean": round(global_mean, 4),
        "date_min": min_ts.date().isoformat(),
        "date_max": max_ts.date().isoformat(),

        "matrix_density_pct": round(
            n_ratings / (n_users * n_movies) * 100, 4
        )
    }])

    save(kpis, "kpis")

    # 2. rating_dist

    counts = np.bincount(
        np
        .floor(r * 2 + 0.5)
        .astype(np.int64),
        minlength=11
    )[1:]

    rating_dist = pd.DataFrame({
        "rating": np.arange(1, 11) / 2,
        "n_ratings": counts
    })

    rating_dist["pct"] = (
        rating_dist["n_ratings"] /
        n_ratings * 100
    ).round(4)

    save(rating_dist, "rating_dist")

    # 3. movie_genre

    m = stats[stats["count"] >= MIN_RATINGS].copy()

    m["mean_rating"] = m["total"] / m["count"]

    m["bayes_rating"] = (
        (PRIOR_C * global_mean + m["total"]) /
        (PRIOR_C + m["count"])
    )

    m["count_tier_order"] = tier_index(
        m["count"],
        MOVIE_FILTER_EDGES
    ) + 1

    m["count_tier"] = (
        m["count_tier_order"]
        .map(dict(enumerate(MOVIE_FILTER_LABELS, 1)))
    )

    # nearest 0.5, half up
    m["mean_bucket"] = np.floor(m["mean_rating"] * 2 + 0.5) / 2

    m = m.rename(columns={
        "year": "release_year",
        "count": "n_ratings"
    })

    m["mean_rating"] = m["mean_rating"].round(4)
    m["bayes_rating"] = m["bayes_rating"].round(4)

    cols = [
        "movie_id",
        "title",
        "release_year",
        "genre",
        "n_ratings",
        "mean_rating",
        "bayes_rating",
        "count_tier",
        "count_tier_order",
        "mean_bucket"
    ]

    by_genre = (
        m.assign(genre=m["genres"].str.split("|"))
        .explode("genre")
        .pipe(
            lambda d: d[~d["genre"].isin(EXCLUDED_GENRES)]
        )
    )

    movie_genre = (
        pd.concat([
            m.assign(genre="All")[cols],
            by_genre[cols]
        ])

        .sort_values(["movie_id", "genre"])
        .reset_index(drop=True)
    )

    save(movie_genre, "movie_genre")

    # 4. genre_stats

    gs = (
        ge.groupby("genre")
        .agg(
            n_ratings=("count", "sum"),
            total=("total", "sum"),
            n_movies=("movie_id", "size")
        )
        .reset_index()
    )

    gs["mean_rating"] = gs["total"] / gs["n_ratings"]
    gs["delta_vs_global"] = (gs["mean_rating"] - global_mean).round(4)
    gs["mean_rating"] = gs["mean_rating"].round(4)
    gs = gs.sort_values("mean_rating", ascending=False)

    save(
        gs[[
            "genre",
            "n_ratings",
            "n_movies",
            "mean_rating",
            "delta_vs_global"
        ]],
        "genre_stats"
    )

    # 5. genre_decade (release decade)

    gd = ge[ge["year"].notna()].copy()

    gd["release_decade"] = (
        (gd["year"].astype(int) // 10 * 10).astype(str)
        + "s"
    )

    gd = (
        gd.groupby(["genre", "release_decade"])
        .agg(
            n_ratings=("count", "sum"),
            total=("total", "sum"),
            n_movies=("movie_id", "size"),
        )
        .reset_index()
    )

    gd = gd[gd["n_ratings"] >= MIN_CELL_RATINGS].copy()
    gd["mean_rating"] = (gd["total"] / gd["n_ratings"]).round(4)

    save(
        gd[[
            "genre",
            "release_decade",
            "n_ratings",
            "n_movies",
            "mean_rating"
        ]],
        "genre_decade"
    )


    # 6. year_stats (year the rating was entered)

    y0 = int(year.min())

    n_y = np.bincount(year - y0)
    t_y = np.bincount(year - y0, weights=r)

    ys = pd.DataFrame({
        "year": y0 + np.arange(len(n_y)),
        "n_ratings": n_y,
        "total": t_y,
    })

    ys = (
        ys[ys["n_ratings"] > 0]
        .reset_index(drop=True)
    )

    ys["mean_rating"] = (
        ys["total"] / ys["n_ratings"]
    ).round(4)

    last = int(ys["year"].max())

    partial = (
        ys["year"].eq(last)
        & (max_ts < pd.Timestamp(year=last, month=12, day=31))
    )

    few = (
        ys["n_ratings"]
        < 0.05 * ys["n_ratings"].median()
    )

    ys["n_partial"] = np.where(
        partial,
        ys["n_ratings"],
        0,
    )

    ys["n_full"] = (
        ys["n_ratings"] - ys["n_partial"]
    )

    ys["note"] = np.where(
        partial,
        "partial year",
        np.where(few, "very few ratings", ""),
    )

    save(
        ys[[
            "year",
            "n_ratings",
            "mean_rating",
            "n_full",
            "n_partial",
            "note",
        ]],
        "year_stats",
    )

    # 7. hour_weekday (UTC)

    cell = np.bincount(
        weekday * 24 + hour,
        minlength=168,
    )

    idx = np.arange(168)

    hw = pd.DataFrame({
        "weekday_num": idx // 24,
        "weekday_name": [
            WEEKDAYS[i] for i in idx // 24
        ],
        "hour_utc": idx % 24,
        "n_ratings": cell,
    })

    save(hw, "hour_weekday")

    # 8. tags_top

    top = (
        tags["tag"]
        .value_counts()
        .head(N_TOP_TAGS)
        .rename_axis("tag")
        .reset_index(name="n_uses")
    )

    top.insert(
        0,
        "rank",
        np.arange(1, len(top) + 1),
    )

    save(top, "tags_top")

    # 9. long_tail

    def tail(entity, per_entity, edges, labels):
        t = pd.DataFrame({"n": per_entity})

        t["tier_order"] = (
            tier_index(t["n"], edges) + 1
        )

        g = (
            t.groupby("tier_order")
            .agg(
                n_entities=("n", "size"),
                n_ratings=("n", "sum"),
            )
            .reset_index()
        )

        g.insert(0, "entity", entity)

        g.insert(
            2,
            "tier_label",
            g["tier_order"].map(
                dict(enumerate(labels, 1))
            ),
        )

        g["pct_entities"] = (
            g["n_entities"]
            / g["n_entities"].sum()
            * 100
        ).round(4)

        g["pct_ratings"] = (
            g["n_ratings"]
            / g["n_ratings"].sum()
            * 100
        ).round(4)

        return g

    lt = pd.concat(
        [
            tail(
                "movie",
                cnt[ids],
                MOVIE_TAIL_EDGES,
                MOVIE_TAIL_LABELS,
            ),
            tail(
                "user",
                u_cnt,
                USER_TAIL_EDGES,
                USER_TAIL_LABELS,
            ),
        ],
        ignore_index=True,
    )

    save(lt, "long_tail")

    # Consistency checks

    assert rating_dist["n_ratings"].sum() == n_ratings
    assert ys["n_ratings"].sum() == n_ratings
    assert hw["n_ratings"].sum() == n_ratings

    assert (
        lt.query("entity == 'movie'")["n_ratings"].sum()
        == n_ratings
    )

    assert (
        lt.query("entity == 'user'")["n_ratings"].sum()
        == n_ratings
    )

    assert (movie_genre["genre"] == "All").sum() == len(m)

    assert not (
        movie_genre
        .groupby(["movie_id", "genre"])
        .size()
        .gt(1)
        .any()
    )

    print("\nAll consistency checks passed.\n")
    print(kpis.T.to_string(header=False))

    print(
        "\nYear flags:",
        ys.loc[
            ys["note"] != "",
            ["year", "note"],
        ].to_dict("records"),
    )

    print("Top 10 by Bayesian average (genre = All):")

    print(
        movie_genre[movie_genre["genre"] == "All"]
        .nlargest(10, "bayes_rating")[
            [
                "title",
                "n_ratings",
                "mean_rating",
                "bayes_rating",
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--clean-dir",
        type=Path,
        default=Path("data/clean/ml-32m"),
    )

    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/agg/ml-32m"),
    )

    a = ap.parse_args()
    main(a.clean_dir, a.out_dir)
