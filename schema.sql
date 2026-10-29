-- HFGCS message database schema (SQLite >= 3.34 for FTS5 trigram; tested on 3.46.1)
-- Proposal by Database & Analytics. See docs/database.md.
--
-- Conventions
--   * All timestamps are UTC, ISO-8601 text: 'YYYY-MM-DDTHH:MM:SSZ' (optionally '.sss' before Z).
--     Text ISO timestamps sort correctly and work with SQLite date functions.
--   * Frequencies are dial (USB carrier) frequencies in kHz, REAL.
--   * Message text keeps NEET INTEL transcription glyphs:
--       '.' = character heard but unreadable, '?' = unknown/garbled,
--       '_' = no message body (callsign-only call), '%' = "no traffic this day" sentinel row.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    note        TEXT
);
INSERT OR IGNORE INTO schema_version(version, note) VALUES (1, 'initial proposal');

-- ---------------------------------------------------------------------------
-- Tunable settings. CLI flags override these; these override built-in defaults.
-- Dedup defaults are provisional: to be tuned against missileer.watch data once
-- Decoding has an evaluation set (see docs/database.md section 3).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS settings (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    description TEXT,
    updated_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
INSERT OR IGNORE INTO settings(key, value, description) VALUES
 ('link.simulcast_sec', '120',  'max |dt| (s) for two copies to be the same emission (simulcast)'),
 ('link.window_hours',  '48',   'max dt (h) for a later copy to be linked as a rebroadcast'),
 ('link.threshold',     '0.90', 'min text similarity (0..1) to link two copies'),
 ('link.min_known',     '0.5',  'min fraction of positions known in both copies (builtin only; decoding fixes 0.5 in code)'),
 ('link.similarity',    'auto', 'auto = decoding.similarity.text_similarity if importable, else builtin; or builtin / decoding'),
 ('link.max_len_diff',  '2',    'max length difference compared (decoding similarity only; builtin compares equal lengths)'),
 ('link.partial_min_len', '12', 'shortest truncated copy (chars known in both) linked to a longer copy as partial; 0 = off');

-- ---------------------------------------------------------------------------
-- Message type taxonomy (lookup table, enforced by FK from messages.msg_type).
-- Type names are informed by eam.watch's public categories (ALLSTATIONS, SKYKING,
-- RADIOCHECK, SKYMASTER, SKYBIRD, DISREGARDED, BACKEND, OTHER) and missileer.watch
-- grades (message, radio-check, auth, traffic, 8888). Ideas only; no code reused.
-- "Disregarded" is modelled as broadcast_type='disregard', not as a type, so a
-- disregarded EAM stays countable as an EAM.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS message_types (
    code            TEXT PRIMARY KEY,
    label           TEXT NOT NULL,
    description     TEXT,
    counts_as_eam   INTEGER NOT NULL DEFAULT 0 CHECK (counts_as_eam IN (0,1)),
    eamwatch_type   TEXT,      -- closest eam.watch category, for import/export mapping
    missileer_grade TEXT       -- closest missileer.watch grade
);
INSERT OR IGNORE INTO message_types(code, label, description, counts_as_eam, eamwatch_type, missileer_grade) VALUES
 ('EAM',        'Emergency Action Message', 'Coded character string to all stations, preamble + body ("message of N characters follows")', 1, 'ALLSTATIONS', 'message'),
 ('SKYKING',    'Skyking',                  'SKYKING broadcast: codeword, TIME nn, AUTHENTICATION xx', 0, 'SKYKING', NULL),
 ('SKYMASTER',  'Skymaster',                'SKYMASTER-addressed traffic/event (see NEET INTEL FAQ)', 0, 'SKYMASTER', NULL),
 ('SKYBIRD',    'Skybird',                  'SKYBIRD-addressed broadcast (legacy collective call); semantics to confirm', 0, 'SKYBIRD', NULL),
 ('EAM_8888',   '8888 message',             'Fixed-head/fixed-tail "8888" format (NEET INTEL FAQ)', 1, NULL, '8888'),
 ('RADIOCHECK', 'Radio check',              'Station or aircraft requesting/answering a radio check', 0, 'RADIOCHECK', 'radio-check'),
 ('AUTH',       'Authentication',           'Authentication challenge/reply exchange', 0, NULL, 'auth'),
 ('TRAFFIC',    'Voice traffic',            'Phone patch or other plain voice traffic', 0, NULL, 'traffic'),
 ('BACKEND',    'Backend',                  'eam.watch category kept for mapping; meaning not documented publicly', 0, 'BACKEND', NULL),
 ('OTHER',      'Other',                    'Anything else (callsign-only call, test count, unknown)', 0, 'OTHER', NULL),
 ('NIL',        'No traffic marker',        'NEET INTEL "%" row: no messages broadcast that day', 0, NULL, NULL);

-- ---------------------------------------------------------------------------
-- External sources and import runs (provenance, licence, attribution).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS external_sources (
    code        TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    url         TEXT,
    license     TEXT,          -- as verified; NULL = none found (reference only)
    attribution TEXT,
    notes       TEXT
);
INSERT OR IGNORE INTO external_sources(code, name, url, license, attribution, notes) VALUES
 ('neetintel',     'NEET INTEL',       'https://neetintel.github.io', NULL, 'NEET INTEL', 'Repos have no LICENSE: column names/ideas only; import their CSVs only with permission'),
 ('missileer',     'missileer.watch',  'https://data.missileer.watch/public/archive.json', 'CC0-1.0', 'attribution appreciated: missileer.watch', 'License field in feed + site footer, verified 2026-10-26'),
 ('warwatchintel', 'War Watch Intel',  'https://www.warwatchintel.com/intel/eams', NULL, 'War Watch Intel', 'No data licence; robots Content-Signal use=reference. Not scraped: daily counts are entered by hand via CSV only'),
 ('eamwatch',      'EAM Watch',        'https://eam.watch', NULL, 'eam.watch', 'Legacy code unlicensed: taxonomy ideas only');

CREATE TABLE IF NOT EXISTS import_runs (
    id            INTEGER PRIMARY KEY,
    source        TEXT NOT NULL REFERENCES external_sources(code),
    started_utc   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    location      TEXT,        -- URL or file path
    etag          TEXT,
    last_modified TEXT,
    generated_at  TEXT,        -- feed's own generation time
    status        TEXT,        -- 'ok', 'not_modified', 'error'
    rows_seen     INTEGER,
    rows_upserted INTEGER,
    notes         TEXT
);

-- ---------------------------------------------------------------------------
-- Receivers: anything that produced audio or a transcription.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS receivers (
    id          INTEGER PRIMARY KEY,
    code        TEXT NOT NULL UNIQUE,              -- short stable id, e.g. 'local-hfplus', 'kiwi-xyz'
    name        TEXT,
    kind        TEXT NOT NULL DEFAULT 'other'
                CHECK (kind IN ('local_sdr','openwebrx','kiwisdr','websdr','livestream','manual','other')),
    url         TEXT,
    latitude    REAL CHECK (latitude  BETWEEN -90  AND 90),
    longitude   REAL CHECK (longitude BETWEEN -180 AND 180),
    maidenhead  TEXT,
    antenna     TEXT,
    active      INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0,1)),
    notes       TEXT,
    created_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- ---------------------------------------------------------------------------
-- Captures: one recorded audio segment (one squelch opening / one file).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS captures (
    id              INTEGER PRIMARY KEY,
    receiver_id     INTEGER NOT NULL REFERENCES receivers(id) ON DELETE RESTRICT,
    frequency_khz   REAL NOT NULL CHECK (frequency_khz > 0),
    mode            TEXT NOT NULL DEFAULT 'USB',
    start_utc       TEXT NOT NULL CHECK (start_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]*Z'),
    end_utc         TEXT CHECK (end_utc IS NULL OR end_utc >= start_utc),
    duration_sec    REAL CHECK (duration_sec IS NULL OR duration_sec >= 0),
    snr_db          REAL,
    noise_floor_db  REAL,
    audio_path      TEXT,                          -- path relative to the recordings root
    audio_url       TEXT,                          -- public/archive link (e.g. stream URL with ?t=)
    audio_sha256    TEXT UNIQUE,                   -- idempotent ingest key for files
    sample_rate_hz  INTEGER,
    voice_start_s         REAL,     -- seconds into the audio where voice starts (missileer voice_start_s)
    voice_start_machine_s REAL,     -- machine estimate kept when a human value replaces it
    voice_start_method    TEXT,     -- e.g. 'vad', 'word_ts_segment', 'ear'
    external_source TEXT REFERENCES external_sources(code),
    external_id     TEXT,
    classification  TEXT CHECK (classification IS NULL OR classification IN ('voice','ale','data','carrier','noise','unknown')),
    source_catalog  TEXT,                          -- e.g. 'hfgcs_catalog.jsonl:123'
    raw_json        TEXT CHECK (raw_json IS NULL OR json_valid(raw_json)),
    notes           TEXT,
    created_utc     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    UNIQUE (external_source, external_id)
);
-- Natural key for our own captures only: external feeds (missileer) report the
-- message's first-recording time per reception, which several clips can share.
CREATE UNIQUE INDEX IF NOT EXISTS uq_captures_own ON captures(receiver_id, frequency_khz, start_utc)
    WHERE external_source IS NULL;
CREATE INDEX IF NOT EXISTS idx_captures_start      ON captures(start_utc);
CREATE INDEX IF NOT EXISTS idx_captures_freq_start ON captures(frequency_khz, start_utc);
CREATE INDEX IF NOT EXISTS idx_captures_receiver   ON captures(receiver_id, start_utc);

-- ---------------------------------------------------------------------------
-- Prefix -> NEET-style behavioural group (G1..G5). Prefixes get reused across
-- years, so the mapping is time-bounded.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prefix_groups (
    id         INTEGER PRIMARY KEY,
    prefix     TEXT NOT NULL,
    group_no   INTEGER NOT NULL CHECK (group_no BETWEEN 1 AND 5),
    valid_from TEXT NOT NULL DEFAULT '0000-01-01',  -- inclusive, YYYY-MM-DD
    valid_to   TEXT NOT NULL DEFAULT '9999-12-31',  -- inclusive
    source     TEXT,
    UNIQUE (prefix, valid_from)
);

-- ---------------------------------------------------------------------------
-- Rebroadcast clusters: one row per distinct message content.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS message_clusters (
    id              INTEGER PRIMARY KEY,
    canonical_text  TEXT,          -- consensus text (best-confidence characters)
    prefix          TEXT,
    length          INTEGER,
    first_heard_utc TEXT,
    last_heard_utc  TEXT,
    n_messages      INTEGER NOT NULL DEFAULT 0,
    n_airings       INTEGER NOT NULL DEFAULT 0,  -- distinct broadcasts (simulcast copies collapsed)
    updated_utc     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- ---------------------------------------------------------------------------
-- Messages: one parsed message as heard on one capture (or one imported log row).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS messages (
    id               INTEGER PRIMARY KEY,
    capture_id       INTEGER REFERENCES captures(id) ON DELETE SET NULL,  -- NULL for imported logs
    receiver_id      INTEGER REFERENCES receivers(id),                    -- denormalised for imports without capture
    frequency_khz    REAL,                                                -- denormalised (NULL when unknown)
    heard_utc        TEXT NOT NULL CHECK (heard_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]*Z'),
    heard_at_basis   TEXT CHECK (heard_at_basis IS NULL OR heard_at_basis IN ('voice_onset','clip_start','log_minute','unknown')),
    offset_sec       REAL,                                                -- start offset inside the capture audio
    msg_type         TEXT NOT NULL DEFAULT 'EAM' REFERENCES message_types(code),
    broadcast_type   TEXT NOT NULL DEFAULT 'broadcast'
                     CHECK (broadcast_type IN ('broadcast','directed','rebroadcast','disregard','callsign_only','test','unknown')),
    callsign         TEXT,                                -- e.g. 'NEW DEAL' (daily rotating) or 'MAINSAIL'
    prefix           TEXT,                                -- 2-char EAM prefix ('PR')
    message_text     TEXT,                                -- full character string, glyphs preserved
    length           INTEGER CHECK (length IS NULL OR length >= 0),   -- declared/logged length ('LN')
    text_length      INTEGER GENERATED ALWAYS AS (length(message_text)) VIRTUAL,
    text_norm        TEXT GENERATED ALWAYS AS (upper(replace(replace(replace(coalesce(message_text,''),' ',''),'-',''),char(9),''))) VIRTUAL,
    preamble         TEXT,                                -- first 6 characters as published (may hold '?')
    preamble_vote    TEXT,                                -- per-position vote across readings/receptions, '?' unknown
    announced_length INTEGER,                             -- spoken "message of N characters follows"
    announced_length_machine INTEGER,                     -- machine value kept when a listener overrides
    group_count      INTEGER,                             -- number of character groups as heard
    partial          INTEGER NOT NULL DEFAULT 0 CHECK (partial IN (0,1)),
    malformed        INTEGER NOT NULL DEFAULT 0 CHECK (malformed IN (0,1)),
    grade            TEXT CHECK (grade IS NULL OR grade IN ('message','unconfirmed','fragment','radio-check','auth','traffic','8888')),
    is_first_reception INTEGER CHECK (is_first_reception IS NULL OR is_first_reception IN (0,1)),
    coreception      INTEGER CHECK (coreception IS NULL OR coreception IN (0,1)),  -- same emission, another receiver
    addressee        TEXT,                                -- 'FOR ANITA 1' -> 'ANITA 1'
    group_no         INTEGER CHECK (group_no IS NULL OR group_no BETWEEN 1 AND 5),  -- explicit override; else prefix_groups
    qualifier        TEXT,                                -- NEET 'Q' glyph: > + & \ * %
    specification    TEXT,                                -- NEET raw SPECIFICATION
    e6_tail          TEXT,                                -- NEET raw E6 (e.g. 'AE0410 (FOR)')
    skyking_codeword TEXT,
    skyking_time     TEXT,                                -- 2-digit minute token as read
    skyking_auth     TEXT,
    confidence       REAL CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    char_confidence  TEXT CHECK (char_confidence IS NULL OR json_valid(char_confidence)),  -- JSON array, one per char
    char_times       TEXT CHECK (char_times IS NULL OR json_valid(char_times)),       -- [[t per char, 1st reading],[2nd reading]] s into clip
    anchors          TEXT CHECK (anchors IS NULL OR json_valid(anchors)),             -- [{kind, t}] standby/message_follows/i_say_again/completes/out
    position_notes   TEXT CHECK (position_notes IS NULL OR json_valid(position_notes)), -- per-position disagreements
    machine_text     TEXT,                                -- machine copy replaced by a correction
    corrected        INTEGER NOT NULL DEFAULT 0 CHECK (corrected IN (0,1)),
    correction_source TEXT,
    ear_queue        INTEGER NOT NULL DEFAULT 0 CHECK (ear_queue IN (0,1)),
    ear_queue_reason TEXT,
    transcript       TEXT,                                -- raw ASR/phonetic transcript
    asr_model        TEXT,
    review_status    TEXT NOT NULL DEFAULT 'auto' CHECK (review_status IN ('auto','needs_review','reviewed','rejected')),
    cluster_id       INTEGER REFERENCES message_clusters(id) ON DELETE SET NULL,
    notes            TEXT,
    source           TEXT,                                -- NEET 'SOURCE'
    set_label        TEXT,                                -- NEET 'SET'
    origin           TEXT NOT NULL DEFAULT 'pipeline' CHECK (origin IN ('pipeline','neet_import','missileer_import','manual')),
    external_source  TEXT REFERENCES external_sources(code),
    external_id      TEXT,                                -- e.g. missileer entry id
    import_key       TEXT UNIQUE,                         -- idempotent re-import key
    created_utc      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    updated_utc      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_messages_heard         ON messages(heard_utc);
CREATE INDEX IF NOT EXISTS idx_messages_prefix_heard  ON messages(prefix, heard_utc);
CREATE INDEX IF NOT EXISTS idx_messages_callsign      ON messages(callsign, heard_utc);
CREATE INDEX IF NOT EXISTS idx_messages_length        ON messages(length, heard_utc);
CREATE INDEX IF NOT EXISTS idx_messages_type          ON messages(msg_type, heard_utc);
CREATE INDEX IF NOT EXISTS idx_messages_capture       ON messages(capture_id);
CREATE INDEX IF NOT EXISTS idx_messages_cluster       ON messages(cluster_id);
CREATE INDEX IF NOT EXISTS idx_messages_review        ON messages(review_status) WHERE review_status <> 'reviewed';
CREATE INDEX IF NOT EXISTS idx_messages_textnorm      ON messages(text_norm);
CREATE INDEX IF NOT EXISTS idx_messages_preamble      ON messages(preamble);
CREATE INDEX IF NOT EXISTS idx_messages_external      ON messages(external_source, external_id);

-- Keep denormalised receiver/frequency in sync with the capture, and bump updated_utc.
CREATE TRIGGER IF NOT EXISTS trg_messages_fill_from_capture
AFTER INSERT ON messages
WHEN NEW.capture_id IS NOT NULL AND (NEW.receiver_id IS NULL OR NEW.frequency_khz IS NULL)
BEGIN
    UPDATE messages SET
        receiver_id   = coalesce(NEW.receiver_id,   (SELECT receiver_id   FROM captures WHERE id = NEW.capture_id)),
        frequency_khz = coalesce(NEW.frequency_khz, (SELECT frequency_khz FROM captures WHERE id = NEW.capture_id))
    WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_messages_touch
AFTER UPDATE OF message_text, callsign, prefix, addressee, notes, review_status, confidence ON messages
BEGIN
    UPDATE messages SET updated_utc = strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = NEW.id;
END;

-- ---------------------------------------------------------------------------
-- Pairwise links between messages with the same content.
--   simulcast   : same airing heard on another receiver and/or frequency (|dt| <= window)
--   rebroadcast : same content aired again later (ground station or E-6B relay)
-- Stored with message_a < message_b (a is the earlier one when kinds are time-ordered).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS message_links (
    id          INTEGER PRIMARY KEY,
    message_a   INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    message_b   INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    link_kind   TEXT NOT NULL CHECK (link_kind IN ('simulcast','rebroadcast','partial','manual')),
    delta_sec   REAL NOT NULL,               -- |heard_utc(b) - heard_utc(a)|
    similarity  REAL NOT NULL CHECK (similarity BETWEEN 0 AND 1),
    method      TEXT NOT NULL,               -- 'builtin:wildcard_hamming:v1', 'decoding:text_similarity', 'external:missileer', 'manual'
    confirmed   INTEGER NOT NULL DEFAULT 0 CHECK (confirmed IN (0,1)),
    created_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    CHECK (message_a < message_b),
    UNIQUE (message_a, message_b)
);
CREATE INDEX IF NOT EXISTS idx_links_b ON message_links(message_b);

-- ---------------------------------------------------------------------------
-- Full-text search (external-content FTS5, trigram tokenizer so substrings of
-- the 30+ char message strings are searchable: MATCH 'QWFO'). Queries need >= 3 chars.
-- ---------------------------------------------------------------------------
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    message_text, callsign, prefix, addressee, notes, transcript,
    content='messages', content_rowid='id', tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS trg_messages_fts_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, message_text, callsign, prefix, addressee, notes, transcript)
    VALUES (NEW.id, NEW.message_text, NEW.callsign, NEW.prefix, NEW.addressee, NEW.notes, NEW.transcript);
END;
CREATE TRIGGER IF NOT EXISTS trg_messages_fts_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, message_text, callsign, prefix, addressee, notes, transcript)
    VALUES ('delete', OLD.id, OLD.message_text, OLD.callsign, OLD.prefix, OLD.addressee, OLD.notes, OLD.transcript);
END;
CREATE TRIGGER IF NOT EXISTS trg_messages_fts_au
AFTER UPDATE OF message_text, callsign, prefix, addressee, notes, transcript ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, message_text, callsign, prefix, addressee, notes, transcript)
    VALUES ('delete', OLD.id, OLD.message_text, OLD.callsign, OLD.prefix, OLD.addressee, OLD.notes, OLD.transcript);
    INSERT INTO messages_fts(rowid, message_text, callsign, prefix, addressee, notes, transcript)
    VALUES (NEW.id, NEW.message_text, NEW.callsign, NEW.prefix, NEW.addressee, NEW.notes, NEW.transcript);
END;

-- ---------------------------------------------------------------------------
-- Views
-- ---------------------------------------------------------------------------
DROP VIEW IF EXISTS v_messages;
CREATE VIEW v_messages AS
SELECT m.id,
       m.heard_utc,
       substr(m.heard_utc, 1, 10)                         AS date_utc,
       CAST(strftime('%H', m.heard_utc) AS INTEGER)       AS hour_utc,
       substr(m.heard_utc,3,2) || substr(m.heard_utc,6,2) || substr(m.heard_utc,9,2) AS yymmdd,
       m.msg_type, m.broadcast_type, m.callsign, m.prefix, m.preamble, m.message_text,
       m.length, m.text_length, m.addressee,
       coalesce(m.group_no, (SELECT pg.group_no FROM prefix_groups pg
                             WHERE pg.prefix = m.prefix
                               AND substr(m.heard_utc,1,10) BETWEEN pg.valid_from AND pg.valid_to
                             ORDER BY pg.valid_from DESC LIMIT 1)) AS group_no,
       m.qualifier, m.e6_tail, m.confidence, m.review_status, m.grade, m.partial,
       m.heard_at_basis, m.announced_length, m.corrected, m.ear_queue,
       coalesce(m.frequency_khz, c.frequency_khz)          AS frequency_khz,
       r.code                                              AS receiver,
       c.snr_db, c.duration_sec,
       coalesce(c.audio_url, c.audio_path)                 AS audio,
       m.cluster_id, m.origin, m.notes
FROM messages m
LEFT JOIN captures  c ON c.id = m.capture_id
LEFT JOIN receivers r ON r.id = coalesce(m.receiver_id, c.receiver_id);

-- ---------------------------------------------------------------------------
-- Coverage comparison against independent daily EAM counts (e.g. warwatchintel).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS external_daily_counts (
    source        TEXT NOT NULL REFERENCES external_sources(code),
    date_utc      TEXT NOT NULL CHECK (date_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
    metric        TEXT NOT NULL DEFAULT 'eams',   -- 'eams' (broadcasts), 'preambles' (distinct), ...
    count         INTEGER NOT NULL CHECK (count >= 0),
    day_boundary  TEXT NOT NULL DEFAULT 'UTC',    -- what the source uses as a day
    collected_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    method        TEXT NOT NULL DEFAULT 'manual' CHECK (method = 'manual'),   -- hand-entered CSV only; no automated collection
    collected_by  TEXT,
    url           TEXT,
    notes         TEXT,
    PRIMARY KEY (source, date_utc, metric)
);

-- Our own daily EAM figures (UTC days), three ways:
--   eams           : DEFAULT. distinct message contents heard that day
--                    (rebroadcasts and simulcast copies collapsed; a content re-aired
--                    on a later day counts once on each day it was heard)
--   eam_broadcasts : broadcasts (simulcast copies of one emission collapsed, rebroadcasts counted)
--   eam_receptions : every EAM row (every receiver, every copy)
DROP VIEW IF EXISTS v_our_daily_eams;
CREATE VIEW v_our_daily_eams AS
SELECT substr(m.heard_utc,1,10) AS date_utc,
       count(DISTINCT coalesce(m.cluster_id, -m.id)) AS eams,
       sum(NOT EXISTS (SELECT 1 FROM message_links l JOIN messages a ON a.id = l.message_a
                       WHERE l.message_b = m.id AND l.link_kind = 'simulcast'
                         AND substr(a.heard_utc,1,10) = substr(m.heard_utc,1,10))) AS eam_broadcasts,
       count(*) AS eam_receptions,
       count(DISTINCT m.receiver_id) AS receivers
FROM messages m JOIN message_types t ON t.code = m.msg_type AND t.counts_as_eam = 1
WHERE m.review_status <> 'rejected' AND coalesce(m.grade,'message') NOT IN ('fragment')
GROUP BY date_utc;

DROP VIEW IF EXISTS v_coverage_comparison;
CREATE VIEW v_coverage_comparison AS
-- Coverage is measured per DISTINCT message (our `eams` vs the external count).
SELECT e.date_utc, e.source, e.metric, e.count AS external_count,
       coalesce(o.eams, 0)           AS our_eams,          -- distinct (basis of coverage_pct)
       coalesce(o.eam_broadcasts, 0) AS our_broadcasts,    -- informational
       coalesce(o.receivers, 0)      AS our_receivers,
       CASE WHEN e.count > 0 THEN round(100.0 * coalesce(o.eams,0) / e.count, 1) END AS coverage_pct,
       coalesce(o.eams,0) - e.count  AS diff,
       e.day_boundary, e.method, e.collected_utc
FROM external_daily_counts e
LEFT JOIN v_our_daily_eams o ON o.date_utc = e.date_utc
WHERE e.metric = 'eams';

-- Daily report. Default figure = distinct messages (rebroadcasts and simulcast copies
-- collapsed: number of different contents heard that UTC day). Per-broadcast is
-- its own column (simulcast copies collapsed, rebroadcasts counted).
DROP VIEW IF EXISTS v_daily_counts;
CREATE VIEW v_daily_counts AS
SELECT d.date_utc,
       d.eams,                     -- distinct EAM contents (default)
       d.eam_broadcasts,
       d.eam_receptions,
       d.receivers,
       (SELECT count(*) FROM v_messages v WHERE v.date_utc = d.date_utc AND v.msg_type = 'SKYKING') AS skykings,
       (SELECT count(DISTINCT coalesce(v.cluster_id, -v.id)) FROM v_messages v
         WHERE v.date_utc = d.date_utc AND v.msg_type = 'EAM' AND v.text_length <> 30) AS non30_eams,
       (SELECT group_concat(DISTINCT v.prefix) FROM v_messages v
         WHERE v.date_utc = d.date_utc AND v.msg_type = 'EAM') AS prefixes
FROM v_our_daily_eams d;

DROP VIEW IF EXISTS v_hourly_heatmap;   -- day-of-week x hour-of-day (UTC)
CREATE VIEW v_hourly_heatmap AS
SELECT CAST(strftime('%w', heard_utc) AS INTEGER) AS dow,   -- 0 = Sunday
       hour_utc,
       count(DISTINCT CASE WHEN msg_type = 'EAM' THEN coalesce(cluster_id, -id) END) AS eams,  -- distinct (default)
       sum(msg_type = 'EAM')                       AS eam_receptions,
       sum(msg_type = 'SKYKING')                   AS skykings,
       count(*)                                    AS messages
FROM v_messages WHERE msg_type <> 'NIL'
GROUP BY dow, hour_utc;

DROP VIEW IF EXISTS v_length_by_month;
CREATE VIEW v_length_by_month AS
SELECT substr(date_utc,1,7) AS month, text_length, group_no, count(*) AS messages
FROM v_messages WHERE msg_type = 'EAM'
GROUP BY month, text_length, group_no;

DROP VIEW IF EXISTS v_prefix_activity;
CREATE VIEW v_prefix_activity AS
SELECT prefix, group_no, min(heard_utc) AS first_heard, max(heard_utc) AS last_heard,
       count(*) AS messages, count(DISTINCT date_utc) AS active_days,
       group_concat(DISTINCT text_length) AS lengths
FROM v_messages WHERE msg_type = 'EAM' AND prefix IS NOT NULL
GROUP BY prefix, group_no;

DROP VIEW IF EXISTS v_rebroadcasts;
CREATE VIEW v_rebroadcasts AS
SELECT mc.id AS cluster_id, mc.prefix, mc.length, mc.canonical_text,
       mc.first_heard_utc, mc.last_heard_utc, mc.n_messages, mc.n_airings,
       round((julianday(mc.last_heard_utc) - julianday(mc.first_heard_utc)) * 24, 2) AS span_hours
FROM message_clusters mc WHERE mc.n_airings > 1;

DROP VIEW IF EXISTS v_receiver_coverage;
CREATE VIEW v_receiver_coverage AS
SELECT r.code AS receiver, c.frequency_khz, substr(c.start_utc,1,10) AS date_utc,
       count(*) AS captures, round(sum(c.duration_sec)/60.0, 1) AS minutes, round(avg(c.snr_db),1) AS avg_snr_db
FROM captures c JOIN receivers r ON r.id = c.receiver_id
GROUP BY r.code, c.frequency_khz, date_utc;

DROP VIEW IF EXISTS v_review_queue;
CREATE VIEW v_review_queue AS
SELECT * FROM v_messages
WHERE review_status IN ('auto','needs_review')
  AND (ear_queue = 1 OR confidence IS NULL OR confidence < 0.9 OR message_text GLOB '*[.?]*'
       OR length <> text_length OR announced_length <> text_length)
ORDER BY heard_utc DESC;

-- Exact NEET INTEL SHORTWAVES.csv column order/format.
DROP VIEW IF EXISTS v_neet_export;
CREATE VIEW v_neet_export AS
SELECT upper(CASE CAST(strftime('%w', m.heard_utc) AS INTEGER)
             WHEN 0 THEN 'SUN' WHEN 1 THEN 'MON' WHEN 2 THEN 'TUE' WHEN 3 THEN 'WED'
             WHEN 4 THEN 'THU' WHEN 5 THEN 'FRI' ELSE 'SAT' END)          AS "DAY",
       replace(substr(m.heard_utc,1,10), '-', '.')                         AS "DATE",
       substr(m.heard_utc,12,5)                                            AS "UTC",
       coalesce(m.callsign,'')                                             AS "CALLSIGN",
       coalesce(m.prefix,'')                                               AS "PR",
       coalesce(m.message_text,'')                                         AS "MESSAGE",
       coalesce(m.qualifier,'')                                            AS "Q",
       coalesce(CAST(m.length AS TEXT), CAST(length(m.message_text) AS TEXT), '') AS "LN",
       coalesce(m.specification,
                CASE WHEN m.addressee IS NOT NULL THEN 'FOR ' || m.addressee
                     WHEN m.broadcast_type = 'disregard' THEN 'DISREGARDED'
                     WHEN m.msg_type = 'EAM' AND length(m.message_text) <> 30 THEN '[' || length(m.message_text) || ' CHAR]'
                     ELSE '' END)                                          AS "SPECIFICATION",
       coalesce(m.e6_tail,'')                                              AS "E6",
       coalesce(m.notes,'')                                                AS "NOTES",
       coalesce(m.source, r.code, '')                                      AS "SOURCE",
       coalesce(m.set_label,'')                                            AS "SET",
       m.id AS _id, m.heard_utc AS _heard_utc
FROM messages m
LEFT JOIN captures  c ON c.id = m.capture_id
LEFT JOIN receivers r ON r.id = coalesce(m.receiver_id, c.receiver_id)
WHERE m.review_status <> 'rejected';

