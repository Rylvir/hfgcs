-- Migration 0003: peak SNR on captures, and machine decodes kept apart from human edits.

-- Recorder catalog v2 reports both snr_db_mean (-> captures.snr_db) and snr_db_peak.
ALTER TABLE captures ADD COLUMN snr_peak_db REAL;
UPDATE captures SET snr_peak_db = json_extract(raw_json, '$.snr_db_peak')
 WHERE raw_json IS NOT NULL AND json_type(raw_json, '$.snr_db_peak') IN ('real', 'integer');

-- Which decoder produced the row's current machine fields, and when they were loaded.
ALTER TABLE messages ADD COLUMN decoder_version TEXT;
ALTER TABLE messages ADD COLUMN decoded_utc TEXT;
-- Decoding output contract v1 (branch decoding-output): per-broadcast identity and timing.
ALTER TABLE messages ADD COLUMN broadcast_id TEXT;        -- '<sha256[:12]>@<start_s, 0.1 s>' as first loaded
ALTER TABLE messages ADD COLUMN start_s REAL;             -- seconds into the recording file
ALTER TABLE messages ADD COLUMN end_s REAL;
ALTER TABLE messages ADD COLUMN callsigns_all TEXT CHECK (callsigns_all IS NULL OR json_valid(callsigns_all));
ALTER TABLE messages ADD COLUMN confidence_detail TEXT
    CHECK (confidence_detail IS NULL OR json_valid(confidence_detail));
CREATE INDEX IF NOT EXISTS idx_messages_capture_start ON messages(capture_id, start_s);

-- Every machine decode loaded by `hfgcs_db.py load-decodes`, one row per (message,
-- decoder version, record content). applied = 0 means the message had human edits or was
-- checked, so its human version was kept and this machine copy was only recorded here
-- (and in messages.machine_text).
CREATE TABLE IF NOT EXISTS message_decodes (
    id              INTEGER PRIMARY KEY,
    message_id      INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    decoder_version TEXT NOT NULL,
    broadcast_id    TEXT,
    decoded_at      TEXT,                -- decoder's own decoded_at (UTC)
    record_sha256   TEXT NOT NULL,       -- hash of the canonical JSON of the parser record
    machine_text    TEXT,
    transcript      TEXT,
    confidence      REAL,
    applied         INTEGER NOT NULL CHECK (applied IN (0,1)),
    source_file     TEXT,
    record_json     TEXT NOT NULL CHECK (json_valid(record_json)),
    loaded_utc      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE (message_id, decoder_version, record_sha256)
);
CREATE INDEX IF NOT EXISTS idx_message_decodes_msg ON message_decodes(message_id, id);
