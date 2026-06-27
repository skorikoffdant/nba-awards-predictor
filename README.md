# NBA Awards Predictor

Machine learning project for predicting NBA postseason award teams, focused on **All-NBA Teams** and **All-Rookie Teams**.

The project contains a full pipeline for collecting player statistics, building historical datasets, training models, running backtests, and generating final predictions for the 2026 NBA season.

## Project overview

The goal of this project is to predict:

* **All-NBA First, Second and Third Team**
* **All-Rookie First and Second Team**

The task is treated as a combination of classification and ranking.
First, the model estimates how likely each player is to be selected. Then the best candidates are selected and assigned to the correct award team.

For All-NBA, the model selects 15 players:

* 5 players for First Team
* 5 players for Second Team
* 5 players for Third Team

For All-Rookie, the model selects 10 players:

* 5 players for First Team
* 5 players for Second Team

## Final model

The final solution uses two different approaches:

```text
All-NBA:    HGB classifier + fixed top-15 team reorderer
All-Rookie: HGB classifier
```

For All-NBA, the base model first selects the top 15 candidates.
Then a second model reorders these players between First, Second and Third Team.

For All-Rookie, the simpler HGB classifier was kept because additional ranking stages did not improve the results consistently.

## Repository structure

```text
.
├── data/
│   ├── raw/                  # downloaded raw statistics and award labels
│   ├── processed/            # prepared datasets used for modeling
│   ├── src/                  # data preparation, training and prediction scripts
│   ├── static/               # small static files required for 2026 prediction
│   └── legacy/               # older experiments and alternative approaches
│
├── models/
│   └── final/                # final trained models and training report
│
├── Daniil_Kavalevich.json    # final prediction output
├── raport.pdf               # project report
├── requirements.txt
└── README.md
```

## Data

The project uses regular season NBA player statistics from several sources.

Main data sources:

* `nba_api` player statistics:

  * Base stats
  * Advanced stats
  * Usage stats
* Historical All-NBA labels
* Historical All-Rookie labels
* Basketball-Reference advanced statistics
* All-Star selections
* Static 2026 eligibility and All-Star roster information

The dataset covers seasons from **2000 to 2026**.

Historical seasons up to 2025 are used for training and validation.
The 2026 season is used as the target season for the final prediction.

## Static 2026 data

The folder below should not be removed:

```text
data/static/
```

It contains small manually prepared files required to reproduce the 2026 prediction correctly.

These files include:

* official 2026 All-Star roster snapshot
* award eligibility overrides
* information related to formal postseason award eligibility

These files do **not** contain the final 2026 All-NBA or All-Rookie results.
They only contain information known before the final award team selection.

## Features

The final feature set is based on several groups of player-level and team-level features.

Main feature groups:

* basic player statistics
* season totals
* previous season performance
* previous All-NBA history
* team-relative role features
* team success indicators
* All-Star reputation features
* award eligibility flags

Examples of used features:

```text
PTS, REB, AST, GP, MIN, TS_PCT, USG_PCT, PIE,
TOTAL_PTS, TOTAL_REB, TOTAL_AST, TOTAL_MIN,
PREV_PTS, PREV_REB, PREV_AST, PREV_ALL_NBA_LABEL,
TEAM_TOTAL_PTS_SHARE, TEAM_TOTAL_AST_SHARE,
IS_ALL_STAR_THIS_SEASON, PREV_ALL_STAR
```

The final model focuses on stable features from `nba_api`, historical player context, team role indicators and All-Star information.

## Installation

Clone the repository:

```bash
git clone https://github.com/skorikoffdant/nba-awards-predictor.git
cd nba-awards-predictor
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

On Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Full pipeline

To rebuild the whole project from scratch, run:

```bash
python data/src/prepare_data.py
python data/src/train_final.py
python data/src/predict_final.py Daniil_Kavalevich.json
```

The full pipeline performs three main steps:

1. Downloads and prepares data.
2. Trains final models.
3. Generates the final JSON prediction file.

## Step 1: prepare data

```bash
python data/src/prepare_data.py
```

This script:

* downloads NBA player statistics
* fetches historical award labels
* prepares Basketball-Reference advanced statistics
* builds processed datasets
* applies static 2026 overrides
* checks the final feature sets

Generated files are saved in:

```text
data/raw/
data/processed/
```

Important processed datasets:

```text
data/processed/player_seasons_labeled.csv
data/processed/all_nba_dataset.csv
data/processed/all_rookie_dataset.csv
```

## Step 2: train final models

```bash
python data/src/train_final.py
```

This script trains the final models:

```text
All-NBA:    HGB classifier + top-15 reorderer
All-Rookie: HGB classifier
```

The trained models are saved in:

```text
models/final/
```

Main output files:

```text
models/final/final_all_nba_hgb_reorderer.joblib
models/final/final_all_rookie_hgb.joblib
models/final/final_train_report.json
```

## Step 3: generate predictions

```bash
python data/src/predict_final.py Daniil_Kavalevich.json
```

This script loads the final models and generates:

```text
Daniil_Kavalevich.json
Daniil_Kavalevich.csv
```

The JSON file contains the final predicted award teams.

## Example output

Example structure of the prediction file:

```json
{
  "season": 2026,
  "all_nba": {
    "first": [
      "Shai Gilgeous-Alexander",
      "Luka Dončić",
      "Nikola Jokić",
      "Kawhi Leonard",
      "Victor Wembanyama"
    ],
    "second": [
      "Jaylen Brown",
      "Cade Cunningham",
      "Donovan Mitchell",
      "Kevin Durant",
      "Jamal Murray"
    ],
    "third": [
      "Jalen Brunson",
      "Tyrese Maxey",
      "Alperen Sengun",
      "Jalen Johnson",
      "Karl-Anthony Towns"
    ]
  },
  "all_rookie": {
    "first": [
      "Cooper Flagg",
      "Kon Knueppel",
      "VJ Edgecombe",
      "Cedric Coward",
      "Maxime Raynaud"
    ],
    "second": [
      "Ace Bailey",
      "Dylan Harper",
      "Jeremiah Fears",
      "Derik Queen",
      "Collin Murray-Boyles"
    ]
  }
}
```

## Evaluation

The project uses historical backtesting with a rolling time split.

For every test season, the model is trained only on previous seasons:

```text
train(t) = seasons 2000 ... t-1
test(t)  = season t
```

The backtest range is:

```text
2010-2025
```

This prevents the model from using future information during evaluation.

## Scoring

The project uses a custom point-based metric.

For each predicted team:

* 1 correct player = 10 points
* 2 correct players = 20 points
* 3 correct players = 40 points
* 4 correct players = 60 points
* 5 correct players = 90 points

Maximum score:

```text
All-NBA:    270 points
All-Rookie: 180 points
Total:      450 points
```

## Backtest results

The best final model achieved:

```text
team_reorderer total score: 260.50 / 450
All-NBA score:              143.88 / 270
All-Rookie score:           116.62 / 180
```

The baseline HGB classifier achieved:

```text
hgb_classifier total score: 256.94 / 450
All-NBA score:              140.31 / 270
All-Rookie score:           116.63 / 180
```

The reorderer improved the final score mainly by improving the assignment of selected All-NBA players to First, Second and Third Team.

## Main scripts

### `prepare_data.py`

Runs the full data preparation pipeline.

```bash
python data/src/prepare_data.py
```

### `train_final.py`

Trains the final models and saves them to `models/final/`.

```bash
python data/src/train_final.py
```

### `predict_final.py`

Generates final predictions in JSON and CSV format.

```bash
python data/src/predict_final.py Daniil_Kavalevich.json
```

### `run_report_experiments.py`

Runs experiments used for model comparison and reporting.

```bash
python data/src/run_report_experiments.py
```

## Reproducibility

The folders below can be regenerated by running the full pipeline:

```text
data/raw/
data/processed/
models/final/
```

However, `data/static/` should be kept because it contains small static files required for the 2026 prediction setup.

## Limitations

This project predicts award selections based on statistical and historical patterns.
It cannot fully model subjective factors such as media narratives, voter preferences, injuries, reputation changes, or late-season context.

The model should be treated as a data-driven prediction system, not as an official award result source.

## Author

Daniil Kavalevich
