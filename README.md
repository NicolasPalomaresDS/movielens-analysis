# MovieLens 32M Analysis

Data cleaning and exploratory analysis project built on the **MovieLens 32M** dataset. It covers downloading and verifying the data, a documented cleaning pipeline, exploratory analysis, and the generation of aggregate tables for a dashboard.

### Data source

[MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) is a movie ratings dataset released in October 2023 by [GroupLens Research](https://grouplens.org/) (University of Minnesota).

| File | Contents |
|---|---|
| `ratings.csv` | 32,000,204 ratings (0.5 to 5.0 in 0.5 steps) from 200,948 users |
| `tags.csv` | 2,000,072 user-applied tags |
| `movies.csv` | 87,585 movies with title and genres |
| `links.csv` | IMDb and TMDb identifiers for each movie |

Every user has at least 20 ratings, and records span January 1995 to October 2023.

> GroupLens' terms of use do not allow redistributing the dataset, so `data/` is not versioned and the data is downloaded locally.

### Project contents

- **`cleaning.ipynb`**: downloads and verifies the data, optimizes dtypes, normalizes column names and text, handles missing values, impossible values and duplicates, and exports to Parquet with a log of the rows removed at each step.
- **`analysis.ipynb`**: Bayesian-average movie ranking, rating distribution, long tail, genres, tags and time patterns.
- **`utils/`**: scripts for downloading the data and building the aggregate tables.

### Reference

F. Maxwell Harper and Joseph A. Konstan. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19, 2015. https://doi.org/10.1145/2827872

## 2. Directory structure

```
movielens/
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- cleaning.ipynb
|-- analysis.ipynb
`-- utils/
    |-- build_aggregates.py
    `-- data_download.py
```

Running the pipeline creates a local `data/` folder:

| Path | Contents |
|---|---|
| `data/raw/ml-32m/` | Original CSVs, `README.txt` and `checksums.txt` |
| `data/clean/ml-32m/` | Clean tables in Parquet and `cleaning_log.csv` |
| `data/agg/ml-32m/` | Aggregate CSVs for the dashboard |

## 3. Reproducing the repository

Requires Python 3.12 and about 1 GB of free disk space for the data.

```bash
git clone https://github.com/NicolasPalomaresDS/movielens-analysis.git
cd movielens

python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter lab
```

Run the notebooks **from the repository root** and in this order:

1. `cleaning.ipynb`: downloads the data, cleans it, and generates `data/clean/` and `data/agg/`.
2. `analysis.ipynb`: reads `data/clean/` and produces the plots and conclusions.