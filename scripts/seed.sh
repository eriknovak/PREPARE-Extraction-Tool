#!/bin/bash
# ============================================================
# seed.sh
# Main seed script for PREPARE Extraction Tool
#
# Prerequisites:
#   cp .env.example .env
#   docker-compose up -d
#   docker-compose exec backend alembic upgrade head
#
# Then:
#   ./scripts/seed.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
POSTGRES_CONTAINER="PREPARE-POSTGRESQL"
ES_CONTAINER="PREPARE-ELASTICSEARCH"

# ------------------------------------------------------------
# Load .env
# ------------------------------------------------------------
if [ -f ".env" ]; then
  set -a
  . ./.env
  set +a
else
  echo "❌ .env file does not exist."
  echo "   First run: cp .env.example .env"
  exit 1
fi

echo "======================================================"
echo "  PREPARE Extraction Tool Data Seeding"
echo "======================================================"
echo "This will import seed data into PostgreSQL and Elasticsearch."
echo

# ------------------------------------------------------------
# Wait for PostgreSQL
# ------------------------------------------------------------
echo "Waiting for PostgreSQL..."
until docker exec "$POSTGRES_CONTAINER" \
  pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" > /dev/null 2>&1
do
  sleep 3
done
echo "✅ PostgreSQL is ready."

# ------------------------------------------------------------
# Check migrations
# ------------------------------------------------------------
echo "Checking if migrations have been executed..."
if ! docker exec -i "$POSTGRES_CONTAINER" \
  psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -tAc "SELECT to_regclass('public.vocabulary');" \
  | grep -q "vocabulary"; then
  echo "❌ Table 'vocabulary' does not exist."
  echo "   First run: docker compose exec backend alembic upgrade head"
  exit 1
fi
echo "✅ Migrations are applied."

# ------------------------------------------------------------
# Wait for Elasticsearch
# ------------------------------------------------------------
echo "Waiting for Elasticsearch..."
until docker exec "$ES_CONTAINER" curl -fsS http://localhost:9200/_cluster/health > /dev/null; do
  sleep 3
done
echo "✅ Elasticsearch is ready."

# ------------------------------------------------------------
# Run PostgreSQL seed
# ------------------------------------------------------------
echo
echo "[1/2] PostgreSQL seed..."
bash "$SCRIPT_DIR/seed_postgres.sh"

# ------------------------------------------------------------
# Run Elasticsearch seed
# ------------------------------------------------------------
echo
echo "[2/2] Elasticsearch seed..."
bash "$SCRIPT_DIR/seed_elasticsearch.sh"

echo
echo "======================================================"
echo "✅ Seeding completed successfully."
echo "======================================================"