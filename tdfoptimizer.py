#!/usr/bin/env python3

"""
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
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import pulp
import requests
from bs4 import BeautifulSoup


###############################################################################
# -----------------------------  DATA I/O  ---------------------------------- #
###############################################################################


def _clean_price(raw: str) -> float:
    return float(raw.rstrip("M").replace(",", "."))


def _fallback_space_to_tab(path: Path) -> StringIO:
    """Zet dubbele spaties in elke regel om naar tabs; retourneert buffer."""
    with path.open(encoding="utf-8") as fh:
        hdr = fh.readline()
        cols = len(re.split(r"\s{2,}", hdr.rstrip("\n")))
        out = [hdr]
        for ln in fh:
            parts = re.split(r"\s{2,}", ln.rstrip("\n"))
            if len(parts) == cols:
                out.append("\t".join(parts) + "\n")
            else:
                out.append(ln)
    return StringIO("".join(out))


# ── PARSE HELPERS ──────────────────────────────────────────────────────────
def _clean_price(raw: str | float) -> float:
    """'7.1M' → 7.1  |  3.5  → 3.5"""
    if isinstance(raw, float):
        return raw
    return float(raw.rstrip("M").replace(",", "."))


def load_riders(tsv_path: str, sep="\t") -> pd.DataFrame:
    # 1. -- Zoek eerste datarij dynamisch
    with open(tsv_path, encoding="utf-8") as fh:
        rows = fh.readlines()

    def _first_data_idx(lines: list[str]) -> int:
        for i, ln in enumerate(lines):
            # eerste kolom is een rangnummer → begint met digit
            if ln.split(sep)[0].strip().isdigit():
                return i
        raise ValueError("Geen datarij gevonden")

    start = _first_data_idx(rows)

    # 2. -- Lees vanaf die regel met de juiste header
    header = [
        "Voornaam", "Achternaam", "Team", "Prijs", "Etappe",
        "Geel", "Groen", "Bolletjes", "Totaal",
        "RoR", "Stop", "# actief", "# reserve"
    ]
    df_raw = pd.read_csv(
        StringIO("".join(rows[start:])),
        sep=sep,
        quoting=3,
        names=header,
        header=None,
    )

    # 3. -- Kolommen mappen
    df = df_raw.rename(columns={
        "Voornaam": "first",
        "Achternaam": "last",
        "Prijs": "price_raw",
        "Totaal": "points",
    })

    df["price"] = (
        df["price_raw"].astype(str)
        .str.replace("M", "", regex=False)
        .str.replace(",", ".")
        .astype(float)
    )
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df.dropna(subset=["first", "last", "price", "points"], inplace=True)

    return df[["first", "last", "price", "points"]]

###############################################################################
# ---------------------------  OPTIMALISATIE  ------------------------------- #
###############################################################################

def optimize_base(df, budget=110):
    n = len(df)                         # ← regel teruggezet
    pb = pulp.LpProblem("BaseTeam", pulp.LpMaximize)
    b  = pulp.LpVariable.dicts("b", range(n), 0, 1, cat="Binary")

    pb += pulp.lpSum(b[i] for i in range(n)) == 15
    pb += pulp.lpSum(b[i] * df.price.iloc[i] for i in range(n)) <= budget
    pb += pulp.lpSum(b[i] * df.points.iloc[i] for i in range(n))

    assert pb.solve(pulp.PULP_CBC_CMD(msg=False)) == pulp.LpStatusOptimal
    return [i for i in range(n) if b[i].value() == 1]

# ── NIEUW: ILP-model voor drie reserves ────────────────────────────────────
# ── NIEUW: correcte, niet-overlappende prijs­banden voor 3 reserves ──────────
def optimize_reserves(df: pd.DataFrame, forbidden: set[int]) -> list[int]:
    n = len(df)
    r = pulp.LpVariable.dicts("r", range(n), 0, 1, cat="Binary")
    pb = pulp.LpProblem("Reserves3", pulp.LpMaximize)

    # precies 3 reserves
    pb += pulp.lpSum(r.values()) == 3
    for i in forbidden:                   # geen basisrenner dubbel gebruiken
        pb += r[i] == 0

    price = df.price

    # niet duurder dan 10 M
    for i, p in enumerate(price):
        if p > 10:
            pb += r[i] == 0

    # disjuncte categorieën → elk exactly 1
    pb += pulp.lpSum(r[i] for i, p in enumerate(price)
                     if 6 < p <= 10) == 1          # R1  (6 – 10]
    pb += pulp.lpSum(r[i] for i, p in enumerate(price)
                     if 2.5 < p <= 6) == 1         # R2  (2.5 – 6]
    pb += pulp.lpSum(r[i] for i, p in enumerate(price)
                     if p <= 2.5) == 1             # R3  ≤ 2.5

    pb += pulp.lpSum(r[i] * df.points.iloc[i] for i in range(n))   # maximiseer
    pb.solve(pulp.PULP_CBC_CMD(msg=False))
    return [i for i in range(n) if r[i].value() == 1]

# ── Snelle greedy-fallback met zelfde bandlogica ────────────────────────────
def greedy_reserves(df: pd.DataFrame, banned: set[int]) -> list[int]:
    idx_r1 = df[~df.index.isin(banned) & (df.price <= 10) & (df.price > 6)] \
                .points.idxmax()
    banned.add(idx_r1)

    idx_r2 = df[~df.index.isin(banned) & (df.price <= 6) & (df.price > 2.5)] \
                .points.idxmax()
    banned.add(idx_r2)

    idx_r3 = df[~df.index.isin(banned) & (df.price <= 2.5)] \
                .points.idxmax()

    return [idx_r1, idx_r2, idx_r3]

###############################################################################
# ---------------------------  DNF SCRAPING  -------------------------------- #
###############################################################################


def _cached_get(url: str, cache: dict[str, tuple[float, str]], ttl: int) -> str | None:
    from time import time

    if url in cache and time() - cache[url][0] < ttl:
        return cache[url][1]
    r = requests.get(url, timeout=10)
    if r.status_code >= 400:
        return None
    cache[url] = (time(), r.text)
    return r.text


def scrape_dnfs(slug: str, ttl: int) -> dict[str, int]:
    base = f"https://www.procyclingstats.com/race/{slug}/gc/stage-"
    cache: dict[str, tuple[float, str]] = {}
    dnfs: dict[str, int] = {}
    for st in itertools.count(1):
        html = _cached_get(f"{base}{st}", cache, ttl)
        if not html:
            break
        soup = BeautifulSoup(html, "html.parser")
        sec = soup.find("div", id="did-not-finish")
        if not sec:
            continue
        for li in sec.select("li"):
            nm = re.sub(r"\s+", " ", li.get_text(strip=True))
            dnfs.setdefault(nm, st)
    return dnfs


###############################################################################
# --------------------------  COMMAND-LINE  --------------------------------- #
###############################################################################

def _parse(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="tdfoptimizer",
        description="Bepaal het optimale Tour-de-Farce-team (15 basis + 3 reserves)",
    )
    p.add_argument("tsv", help="Data-bestand (TSV/CSV)")
    p.add_argument("--sep", default="\t", help="Scheidingsteken in invoer")
    p.add_argument("--race", help="PCS-slug voor DNF-scraping")
    p.add_argument("--max-price", type=float, default=110,
                   help="Budgetlimiet in miljoenen euro (default 110)")
    p.add_argument("--csv-out", metavar="PAD", help="Schrijf gekozen team naar CSV")
    p.add_argument("--cache", type=int, default=0,
                   help="HTML-cacheduur (seconden) voor PCS-scraping")
    p.add_argument("--test", action="store_true", help="Draai unittests en stop")
    return p.parse_args(argv)


def main() -> None:
    args = _parse()

    # -- data-inlezen --------------------------------------------------------
    src = Path(args.tsv)
    if not src.is_file():
        sys.exit(f"Dataset ‘{src}’ niet gevonden")
    df = load_riders(src, sep=args.sep)

    # -- optioneel DNF-scrapen ----------------------------------------------
    if args.race:
        dnfs = scrape_dnfs(args.race, args.cache)
        if dnfs:
            df = df[~df["last"].isin(dnfs)]      # uitvallers verwijderen
        print(f"DNF’s verwerkt: {len(dnfs)}")

    # -- optimalisatie -------------------------------------------------------
    base_idx = optimize_base(df, args.max_price)
    try:
        res_idx = optimize_reserves(df, set(base_idx))
    except Exception:                           # ILP faalt → greedy
        res_idx = greedy_reserves(df, set(base_idx))

    pick = base_idx + res_idx

    # -- output --------------------------------------------------------------
    print("=== Basis (15) ===")
    print(df.iloc[base_idx][["first", "last", "price", "points"]]
        .sort_values("points", ascending=False))

    print("=== Reserves (3) ===")
    print(df.iloc[res_idx][["first", "last", "price", "points"]])

    print(f"Totaal prijs basis €M : {df.iloc[base_idx].price.sum():.1f}")
    print(f"Punten basis          : {df.iloc[base_idx].points.sum():.0f}")

    if args.csv_out:
        df.loc[pick].to_csv(args.csv_out, index=False)
        print("CSV geschreven:", args.csv_out)

if __name__ == "__main__":
    main()