#!/bin/bash

# Gebruik: ./run_team.sh <bestandsnaam.tsv> <koersslug>
# Voorbeeld: ./run_team.sh vuelta24.tsv vuelta-a-espana/2024

FILE="$1"
SLUG="$2"
BUDGET=110
CACHE=3600
OUT="team_${FILE%.tsv}.csv"

if [[ ! -f "$FILE" ]]; then
  echo "❌ Bestand $FILE bestaat niet."
  exit 1
fi

python3 tdfoptimizer.py "$FILE" \
  --race "$SLUG" \
  --max-price "$BUDGET" \
  --csv-out "$OUT" \
  --cache "$CACHE"