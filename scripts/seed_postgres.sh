#!/bin/bash
# ============================================================
# seed_postgres.sh
# Seed PostgreSQL with vocabulary + concept CSV files.
# Preserves original vocabulary.id and concept.id.
# ============================================================

set -euo pipefail

CONTAINER="PREPARE-POSTGRESQL"
DB="${POSTGRES_DB:-}"
USER="${POSTGRES_USER:-}"
SEED_DIR="$(cd "$(dirname "$0")/../seed_data" && pwd)"

if [ -z "$DB" ] || [ -z "$USER" ]; then
  echo "❌ POSTGRES_DB or POSTGRES_USER is not set."
  exit 1
fi

if [ ! -f "$SEED_DIR/vocabulary.csv" ] || [ ! -f "$SEED_DIR/concept.csv" ]; then
  echo "❌ Missing vocabulary.csv or concept.csv in $SEED_DIR"
  exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "❌ Container $CONTAINER is not running."
  exit 1
fi

echo "Starting PostgreSQL seed..."

docker cp "$SEED_DIR/vocabulary.csv" "$CONTAINER:/tmp/vocabulary.csv"
docker cp "$SEED_DIR/concept.csv" "$CONTAINER:/tmp/concept.csv"

docker exec -i "$CONTAINER" psql -v ON_ERROR_STOP=1 -U "$USER" -d "$DB" <<'EOF'
BEGIN;

CREATE TEMP TABLE tmp_vocabulary (
  id             integer,
  name           varchar,
  uploaded       timestamp,
  status         processingstatus,
  error_message  varchar,
  user_id        integer
);

CREATE TEMP TABLE tmp_concept (
  id                integer,
  vocab_term_id     varchar,
  vocab_term_name   varchar,
  domain_id         varchar,
  concept_class_id  varchar,
  standard_concept  varchar,
  concept_code      varchar,
  valid_start_date  timestamp,
  valid_end_date    timestamp,
  invalid_reason    varchar,
  vocabulary_id     integer
);

COPY tmp_vocabulary (id, name, uploaded, status, error_message, user_id)
FROM '/tmp/vocabulary.csv' WITH CSV HEADER;

COPY tmp_concept (
  id, vocab_term_id, vocab_term_name, domain_id, concept_class_id,
  standard_concept, concept_code, valid_start_date, valid_end_date,
  invalid_reason, vocabulary_id
)
FROM '/tmp/concept.csv' WITH CSV HEADER;

INSERT INTO "user" (username, hashed_password, disabled, created_at)
VALUES ('seed_system', 'not-a-real-password', true, CURRENT_TIMESTAMP)
ON CONFLICT (username) DO NOTHING;

INSERT INTO vocabulary (id, name, uploaded, status, error_message, user_id)
SELECT
  tv.id,
  tv.name,
  CURRENT_TIMESTAMP,
  tv.status,
  tv.error_message,
  (SELECT id FROM "user" WHERE username = 'seed_system')
FROM tmp_vocabulary tv
ON CONFLICT (id) DO NOTHING;

-- Safety check: every concept vocabulary_id must exist in vocabulary
DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM tmp_concept tc
    LEFT JOIN vocabulary v ON v.id = tc.vocabulary_id
    WHERE v.id IS NULL
  ) THEN
    RAISE EXCEPTION 'Some concept rows reference missing vocabulary_id values.';
  END IF;
END $$;

INSERT INTO concept (
  id, vocab_term_id, vocab_term_name, domain_id, concept_class_id,
  standard_concept, concept_code, valid_start_date, valid_end_date,
  invalid_reason, vocabulary_id
)
SELECT
  tc.id,
  tc.vocab_term_id,
  tc.vocab_term_name,
  tc.domain_id,
  tc.concept_class_id,
  tc.standard_concept,
  tc.concept_code,
  tc.valid_start_date,
  tc.valid_end_date,
  tc.invalid_reason,
  tc.vocabulary_id
FROM tmp_concept tc
ON CONFLICT (id) DO NOTHING;

SELECT setval(
  pg_get_serial_sequence('vocabulary', 'id'),
  COALESCE((SELECT MAX(id) FROM vocabulary), 1),
  true
)
WHERE pg_get_serial_sequence('vocabulary', 'id') IS NOT NULL;

SELECT setval(
  pg_get_serial_sequence('concept', 'id'),
  COALESCE((SELECT MAX(id) FROM concept), 1),
  true
)
WHERE pg_get_serial_sequence('concept', 'id') IS NOT NULL;

COMMIT;
EOF

docker exec "$CONTAINER" rm -f /tmp/vocabulary.csv /tmp/concept.csv

echo "✅ PostgreSQL seed completed successfully!"