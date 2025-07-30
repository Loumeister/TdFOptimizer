#!/usr/bin/env python3

"""
─────────────────────────  USER GUIDE  ─────────────────────────
1. Install dependencies
   $ pip install pandas pulp beautifulsoup4 requests

2. Export the complete Tour-de-Farce table to a **TSV**
   (keep column headers; separator = tab).

3. Basic run – selects the optimal team (15 + 3 reserves)
   $ python tdfoptimizer.py <DATA.tsv>

4. Enable DNF processing via ProCyclingStats
   $ python tdfoptimizer.py <DATA.tsv> \
       --race giro-d-italia/2025

5. Options
   --sep <str>          input separator (default: tab)
   --max-price <float>  budget limit (default: 110)
   --csv-out <path>     write chosen team to CSV
   --cache <sec>        HTML cache duration for PCS scraping (0 = off)
   --race <slug>        activates DNF scraper; slug = part of PCS URL
   --test               run built-in unit tests and exit

Examples
-----------
# Determine optimal team only
$ python tdfoptimizer.py giro2025.tsv

# Determine team with lower budget
$ python tdfoptimizer.py giro2025.tsv --max-price 105

# CSV export of result and PCS scrape with caching
$ python tdfoptimizer.py giro2025.tsv --race giro-d-italia/2025 \

────────────────────────  GEBRUIKSHANDLEIDING  ────────────────────────
1. Installeer afhankelijkheden
   $ pip install pandas pulp beautifulsoup4 requests

2. Exporteer de volledige Tour-de-Farce-tabel naar een **TSV**  
   (bewaar kolomkoppen; scheidingsteken = tab).

3. Basisrun – selecteert het optimale team (15 + 3 reserves)
   $ python tdfoptimizer.py <DATA.tsv>

4. DNF-verwerking inschakelen via ProCyclingStats
   $ python tdfoptimizer.py <DATA.tsv> \
       --race giro-d-italia/2025

5. Opties
   --sep <str>          scheidingsteken in invoer (standaard: tab)
   --max-price <float>  budget-limiet (default: 110)
   --csv-out <pad>      schrijf gekozen team naar CSV
   --cache <sec>        HTML-cacheduur voor PCS-scraping (0 = uit)
   --race <slug>        activeert DNF-scraper; slug = deel uit PCS-URL
   --test               draai ingebouwde unittests en stop

Voorbeelden
-----------
# Alleen optimaal team bepalen
$ python giro_team_optimizer.py giro2025.tsv

# Team bepalen met lager budget
$ python giro_team_optimizer.py giro2025.tsv --max-price 105

# CSV-export van resultaat en PCS-scrape met caching
$ python giro_team_optimizer.py giro2025.tsv --race giro-d-italia/2025 \

    --csv-out myteam.csv --cache 3600
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
import unittest
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import pulp
import requests
from bs4 import BeautifulSoup

# Number of stages in the race (used for DNF replacement calculations). Adjust if necessary.
TOTAL_STAGES: int = 21


###############################################################################
# -----------------------------  DATA I/O  ---------------------------------- #
###############################################################################


def _clean_price(raw: str | float) -> float:
    """'7.1M' → 7.1  |  3.5  → 3.5"""
    if isinstance(raw, float):
        return raw
    return float(raw.rstrip("M").replace(",", "."))


def _find_data_start_index(rows: list[str], sep: str) -> int:
    """Dynamically find the first data row."""
    for i, ln in enumerate(rows):
        # first column is a rank number → starts with a digit
        if ln.split(sep)[0].strip().isdigit():
            return i
    raise ValueError("No data row found")


def _read_and_clean_data(
    file_content_str: str, sep: str, header: list[str]
) -> pd.DataFrame:
        """Read CSV data, rename columns, convert types, and drop NaNs."""
        df_raw = pd.read_csv(
            StringIO(file_content_str),
            sep=sep,
            quoting=3,
            names=header,
            header=None,
        )

        # Map columns
        df = df_raw.rename(
            columns={
                "FirstName": "first",  # Original column name was "Voornaam"
                "LastName": "last",  # Original column name was "Achternaam"
                "Price": "price_raw",  # Original column name was "Prijs"
                "Total": "points",  # Original column name was "Totaal"
            }
        )

        df["price"] = (
            df["price_raw"]
            .astype(str)
            .str.replace("M", "", regex=False)
            .str.replace(",", ".")
            .astype(float)
        )
        df["points"] = pd.to_numeric(df["points"], errors="coerce")

        # Convert Stop column to numeric stage count if available. Riders that did not stop get NaN.
        if "Stop" in df.columns:
            df["Stop"] = pd.to_numeric(df["Stop"], errors="coerce")

        # Ensure we keep rows with essential data
        df.dropna(subset=["first", "last", "price", "points"], inplace=True)

        # Return columns including Stop if available
        cols = ["first", "last", "price", "points"]
        if "Stop" in df.columns:
            cols.append("Stop")
        return df[cols]


def load_riders(tsv_path: str, sep="\t") -> pd.DataFrame:
    with open(tsv_path, encoding="utf-8") as fh:
        rows = fh.readlines()

    start_index = _find_data_start_index(rows, sep)

    # These are the expected column names in the TSV file
    header = [
        "FirstName",
        "LastName",
        "Team",
        "Price",
        "Stage",
        "Yellow",
        "Green",
        "PolkaDot",
        "Total",
        "RoR",
        "Stop",
        "# active",
        "# reserve",
    ]

    file_content_str = "".join(rows[start_index:])
    return _read_and_clean_data(file_content_str, sep, header)


###############################################################################
# ---------------------------  OPTIMIZATION  -------------------------------- #
###############################################################################


def optimize_base(df, budget=110):
    n = len(df)
    pb = pulp.LpProblem("BaseTeam", pulp.LpMaximize)
    b = pulp.LpVariable.dicts("b", range(n), 0, 1, cat="Binary")

    pb += pulp.lpSum(b[i] for i in range(n)) == 15
    pb += pulp.lpSum(b[i] * df.price.iloc[i] for i in range(n)) <= budget
    # Use 'effective_points' if computed, otherwise fall back to 'points'.
    score_col = df["effective_points"] if "effective_points" in df.columns else df["points"]
    pb += pulp.lpSum(b[i] * score_col.iloc[i] for i in range(n))

    assert pb.solve(pulp.PULP_CBC_CMD(msg=False)) == pulp.LpStatusOptimal
    return [i for i in range(n) if b[i].value() == 1]


def optimize_reserves(df: pd.DataFrame, forbidden: set[int]) -> list[int]:
    n = len(df)
    r = pulp.LpVariable.dicts("r", range(n), 0, 1, cat="Binary")
    pb = pulp.LpProblem("Reserves3", pulp.LpMaximize)

    # exactly 3 reserves
    pb += pulp.lpSum(r.values()) == 3
    for i in forbidden:  # do not reuse base team riders
        pb += r[i] == 0

    price = df.price

    # not more expensive than 10M
    for i, p in enumerate(price):
        if p > 10:
            pb += r[i] == 0

    # disjoint categories → each exactly 1
    pb += pulp.lpSum(r[i] for i, p in enumerate(price) if 6 < p <= 10) == 1  # R1 (6–10]
    pb += pulp.lpSum(r[i] for i, p in enumerate(price) if 2.5 < p <= 6) == 1  # R2 (2.5–6]
    pb += pulp.lpSum(r[i] for i, p in enumerate(price) if p <= 2.5) == 1  # R3 ≤ 2.5

    pb += pulp.lpSum(r[i] * df.points.iloc[i] for i in range(n))  # maximize
    status = pb.solve(pulp.PULP_CBC_CMD(msg=False))
    if status != pulp.LpStatusOptimal:
        raise ValueError(
            f"PuLP could not find an optimal solution for reserves. Status: {pulp.LpStatus[status]}"
        )
    return [i for i in range(n) if r[i].value() == 1]


def greedy_reserves(df: pd.DataFrame, banned: set[int]) -> list[int]:
    idx_r1 = (
        df[~df.index.isin(banned) & (df.price <= 10) & (df.price > 6)].points.idxmax()
    )
    banned.add(idx_r1)

    idx_r2 = (
        df[~df.index.isin(banned) & (df.price <= 6) & (df.price > 2.5)]
        .points.idxmax()
    )
    banned.add(idx_r2)

    idx_r3 = df[~df.index.isin(banned) & (df.price <= 2.5)].points.idxmax()

    return [idx_r1, idx_r2, idx_r3]


def compute_effective_points(df: pd.DataFrame, total_stages: int = TOTAL_STAGES) -> pd.DataFrame:
    """
    Compute an approximate 'effective_points' for each rider that takes into
    account the possibility of being replaced by a reserve after a DNF.

    The fantasy competition allows a reserve to substitute for a rider who does
    not start or abandons the race. A substitute starts scoring points from the
    stage after the base rider stops. Since the provided input only contains a
    rider's total points for the whole race (or up to the stage they stopped),
    this function approximates the replacement logic by assuming points accrue
    evenly across stages. The effective points for a rider who stops early are:

        points + (remaining_stages / total_stages) * best_reserve_points

    where `best_reserve_points` is the maximum points scored by any rider
    eligible to serve as the reserve for this price category.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least 'price', 'points' and optionally 'Stop'
        columns.
    total_stages : int, optional
        Total number of stages in the race (default TOTAL_STAGES).

    Returns
    -------
    pd.DataFrame
        DataFrame with an added 'effective_points' column.
    """
    # Determine the highest scoring rider within each reserve price cap.
    # Reserves have maximum prices of 10, 6 and 2.5 million euros respectively.
    # Use original points here so reserves are selected on their own merits.
    p_r1_max = df.loc[df["price"] <= 10, "points"].max()
    p_r2_max = df.loc[df["price"] <= 6, "points"].max()
    p_r3_max = df.loc[df["price"] <= 2.5, "points"].max()
    # Fill missing maxima with zero if no riders fit in that category.
    p_r1_max = p_r1_max if pd.notna(p_r1_max) else 0.0
    p_r2_max = p_r2_max if pd.notna(p_r2_max) else 0.0
    p_r3_max = p_r3_max if pd.notna(p_r3_max) else 0.0

    def best_reserve_points(price: float) -> float:
        """Return the best reserve points for a given base rider price."""
        if price >= 10:
            return p_r1_max
        elif price > 6:
            return p_r2_max
        elif price > 2.5:
            return p_r3_max
        else:
            # No cheaper reserve exists for riders costing <= 2.5M
            return 0.0

    eff_points: list[float] = []
    # Iterate rows to compute effective points
    for _, row in df.iterrows():
        pts = row["points"]
        # Stop stage if available; default to total_stages for riders who finish.
        stop_stage = None
        if "Stop" in df.columns:
            stop_val = row.get("Stop")
            stop_stage = stop_val if pd.notna(stop_val) else total_stages
        else:
            stop_stage = total_stages
        # Ensure stop_stage is numeric and within sensible bounds.
        if stop_stage is None or pd.isna(stop_stage) or stop_stage > total_stages or stop_stage < 0:
            stop_stage = total_stages
        # Non-starter or early DNF: approximate remaining contribution from a reserve.
        if stop_stage < total_stages:
            remaining_points = (total_stages - stop_stage) / total_stages * best_reserve_points(row["price"])
            eff_points.append(float(pts) + remaining_points)
        else:
            eff_points.append(float(pts))
    df = df.copy()
    df["effective_points"] = eff_points
    return df


###############################################################################
# ---------------------------  DNF SCRAPING  -------------------------------- #
###############################################################################


def _cached_get(url: str, cache: dict[str, tuple[float, str]], ttl: int) -> str | None:
    from time import time

    if url in cache and time() - cache[url][0] < ttl:
        return cache[url][1]
    try:
        r = requests.get(url, timeout=10)
        if r.status_code >= 400:
            print(
                f"Warning: Failed to fetch {url}. Status code: {r.status_code}",
                file=sys.stderr,
            )
            return None
        cache[url] = (time(), r.text)
        return r.text
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to fetch {url}. Error: {e}", file=sys.stderr)
        return None


def scrape_dnfs(slug: str, ttl: int) -> dict[str, int]:
    base = f"https://www.procyclingstats.com/race/{slug}/gc/stage-"
    cache: dict[str, tuple[float, str]] = {}
    dnfs: dict[str, int] = {}
    for st in itertools.count(1):
        stage_url = f"{base}{st}"
        html = _cached_get(stage_url, cache, ttl)
        if not html:
            break  # Stop if a stage page isn't found (likely end of race)

        try:
            soup = BeautifulSoup(html, "html.parser")
            sec = soup.find("div", id="did-not-finish")
            if not sec:
                # This can be normal if no DNFs for a stage, or if page structure changed
                print(
                    f"Info: No DNF section found for stage {st}. URL: {stage_url}",
                    file=sys.stderr,
                )
                continue
            for li in sec.select("li"):
                nm = re.sub(r"\s+", " ", li.get_text(strip=True))
                dnfs.setdefault(nm, st)
        except Exception as e:
            print(
                f"Error: Failed to parse DNF data for stage {st}. URL: {stage_url}. Error: {e}",
                file=sys.stderr,
            )
            continue  # Try next stage even if current one fails
    return dnfs


###############################################################################
# --------------------------  COMMAND-LINE  --------------------------------- #
###############################################################################


def _parse(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="tdfoptimizer",
        description="Determine the optimal Tour-de-Farce team (15 base + 3 reserves)",
    )
    p.add_argument("tsv", help="Data file (TSV/CSV)")
    p.add_argument("--sep", default="\t", help="Input separator")
    p.add_argument("--race", help="PCS slug for DNF scraping")
    p.add_argument(
        "--max-price",
        type=float,
        default=110,
        help="Budget limit in millions of euros (default 110)",
    )
    p.add_argument("--csv-out", metavar="PATH", help="Write chosen team to CSV")
    p.add_argument(
        "--cache",
        type=int,
        default=0,
        help="HTML cache duration (seconds) for PCS scraping",
    )
    p.add_argument("--test", action="store_true", help="Run unit tests and exit")

    return p.parse_args(argv)


def main() -> None:
    args = _parse()

    if args.test:
        print("Running unit tests...")
        # Pass only the script name to unittest.main to avoid conflicts with argparse
        unittest.main(argv=[sys.argv[0]], verbosity=1, exit=True)

    # -- data loading --------------------------------------------------------
    src = Path(args.tsv)
    if not src.is_file():
        raise FileNotFoundError(f"Dataset '{src}' not found")
    df = load_riders(src, sep=args.sep)

    # -- optional DNF scraping ----------------------------------------------
    if args.race:
        dnfs = scrape_dnfs(args.race, args.cache)
        if dnfs:
            # Update stop stages instead of removing riders. If a rider already has a
            # recorded stop stage in the data, keep the earliest (minimum) of the
            # two. scrape_dnfs returns a mapping of rider names to their DNF stage.
            for nm, st in dnfs.items():
                # Normalise the scraped name for comparison (case-insensitive)
                nm_normalised = nm.strip().lower()
                # Match on full name (first + last) where possible
                full_name_series = (
                    df["first"].astype(str).str.strip().str.lower() + " " + df["last"].astype(str).str.strip().str.lower()
                )
                mask_full = full_name_series == nm_normalised
                # Fallback: match on last name only if no full match found
                if not mask_full.any():
                    last_only = nm_normalised.split()[-1]
                    mask_full = df["last"].astype(str).str.strip().str.lower() == last_only
                # Update the Stop column: if Stop exists, take the minimum of existing and new
                if mask_full.any():
                    # Ensure 'Stop' column exists
                    if "Stop" not in df.columns:
                        df["Stop"] = float("nan")
                    for idx in df.index[mask_full]:
                        current = df.at[idx, "Stop"]
                        if pd.isna(current) or st < current:
                            df.at[idx, "Stop"] = float(st)
            print(f"DNFs processed: {len(dnfs)}")
        else:
            print("DNFs processed: 0")

    # Compute effective points based on stop stages. Always compute this after
    # scraping DNFs so that any scraped DNF data is incorporated. If the
    # dataset does not include a 'Stop' column, we still add 'effective_points'
    # equal to 'points'.
    if "Stop" in df.columns:
        df = compute_effective_points(df, TOTAL_STAGES)
    else:
        df["effective_points"] = df["points"].astype(float)

    # -- optimization -------------------------------------------------------
    base_idx = optimize_base(df, args.max_price)
    try:
        res_idx = optimize_reserves(df, set(base_idx))
    except ValueError as e:  # Catching specific ValueError from optimize_reserves
        print(
            f"Warning: ILP optimization for reserves failed: {e}. Falling back to greedy algorithm.",
            file=sys.stderr,
        )
        res_idx = greedy_reserves(df, set(base_idx))
    except pulp.PulpSolverError as e:  # Catching PuLP specific solver errors
        print(
            f"Warning: PuLP solver error during reserve optimization: {e}. Falling back to greedy algorithm.",
            file=sys.stderr,
        )
        res_idx = greedy_reserves(df, set(base_idx))

    pick = base_idx + res_idx

    # -- output --------------------------------------------------------------

    print("=== Base Team (15) ===")
    print(
        df.iloc[base_idx][["first", "last", "price", "points"]].sort_values(
            "points", ascending=False
        )
    )

    print("=== Reserves (3) ===")
    print(df.iloc[res_idx][["first", "last", "price", "points"]])

    print(f"Total price base team €M : {df.iloc[base_idx].price.sum():.1f}")
    print(f"Base team points       : {df.iloc[base_idx].points.sum():.0f}")

    if args.csv_out:
        df.loc[pick].to_csv(args.csv_out, index=False)
        print("CSV written:", args.csv_out)


class TestOptimizer(unittest.TestCase):
    def test_clean_price(self):
        self.assertEqual(_clean_price("7.1M"), 7.1)
        self.assertEqual(_clean_price("3,5M"), 3.5)  # Handles comma as decimal separator
        self.assertEqual(_clean_price(5.0), 5.0)  # Handles float input
        self.assertAlmostEqual(_clean_price("10M"), 10.0)  # Handles "M" suffix


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:  # Catching other ValueErrors that might be raised from main/helpers
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


class TestOptimizer(unittest.TestCase):
    def test_clean_price(self):
        self.assertEqual(_clean_price("7.1M"), 7.1)
        self.assertEqual(_clean_price("3,5M"), 3.5)  # Handles comma as decimal separator
        self.assertEqual(_clean_price(5.0), 5.0)  # Handles float input
        self.assertAlmostEqual(_clean_price("10M"), 10.0)  # Handles "M" suffix


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:  # Catching other ValueErrors that might be raised from main/helpers
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)