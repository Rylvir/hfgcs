-- Migration 0002: human review of decoded messages (review app, docs/review-ui.md).
-- Applied by `scripts/hfgcs_db.py migrate` (and by `init`) through hfgcs/db_migrations.py,
-- which records version 2 in schema_version. Only adds objects, so it is safe on a v1 DB.

-- One row per reviewed message. checked = 1 means a person listened and confirmed the
-- message's transcript and parsed fields; such rows are exported as gold labels.
CREATE TABLE IF NOT EXISTS reviews (
    message_id    INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
    checked       INTEGER NOT NULL DEFAULT 0 CHECK (checked IN (0,1)),
    notes         TEXT,
    reviewer      TEXT,                 -- who last changed checked/notes
    reviewed_utc  TEXT CHECK (reviewed_utc IS NULL OR reviewed_utc GLOB
                    '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]*Z'),
    created_utc   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_utc   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_reviews_checked ON reviews(checked, reviewed_utc);

-- Append-only audit trail: one row per changed field. No foreign key on message_id so
-- history survives if a message is later deleted or re-imported.
CREATE TABLE IF NOT EXISTS edit_history (
    id            INTEGER PRIMARY KEY,
    edited_utc    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    editor        TEXT NOT NULL CHECK (length(editor) BETWEEN 1 AND 200),
    editor_source TEXT NOT NULL CHECK (editor_source IN ('cf_access','form','default','cli')),
    change_id     TEXT NOT NULL,        -- groups the fields saved by one form submission
    table_name    TEXT NOT NULL CHECK (table_name IN ('messages','reviews')),
    message_id    INTEGER NOT NULL,
    field         TEXT NOT NULL,
    old_value     TEXT,                 -- NULL = was empty/absent
    new_value     TEXT
);
CREATE INDEX IF NOT EXISTS idx_edit_history_msg ON edit_history(message_id, id);

CREATE TRIGGER IF NOT EXISTS trg_edit_history_no_update BEFORE UPDATE ON edit_history
BEGIN SELECT RAISE(ABORT, 'edit_history is append-only'); END;
CREATE TRIGGER IF NOT EXISTS trg_edit_history_no_delete BEFORE DELETE ON edit_history
BEGIN SELECT RAISE(ABORT, 'edit_history is append-only'); END;

-- Gold labels = checked reviews joined to their message and capture.
DROP VIEW IF EXISTS v_gold_labels;
CREATE VIEW v_gold_labels AS
SELECT m.id AS message_id, m.msg_type, m.heard_utc, m.frequency_khz, m.callsign, m.prefix,
       m.message_text, m.text_norm, m.length, m.announced_length, m.preamble, m.addressee,
       m.transcript, m.machine_text, m.corrected, m.confidence,
       c.id AS capture_id, c.audio_path, c.audio_sha256, c.voice_start_s, c.start_utc AS capture_start_utc,
       rc.code AS receiver, r.reviewer, r.reviewed_utc, r.notes AS review_notes
FROM reviews r
JOIN messages m ON m.id = r.message_id
LEFT JOIN captures c ON c.id = m.capture_id
LEFT JOIN receivers rc ON rc.id = coalesce(m.receiver_id, c.receiver_id)
WHERE r.checked = 1;
