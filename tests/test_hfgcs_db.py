"""Stdlib tests for schema.sql and scripts/hfgcs_db.py. All data is fake. Run:
    python3 -m unittest discover -s tests -v
"""
import contextlib
import io
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "scripts"))
import hfgcs_db  # noqa: E402


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        rc = hfgcs_db.main(list(argv))
    return rc, out.getvalue()


class DbTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "t.db")
        self.assertEqual(run("--db", self.db, "init")[0], 0)
        self.con = sqlite3.connect(self.db)
        self.con.execute("PRAGMA foreign_keys = ON")   # per-connection in SQLite

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def q(self, sql, *p):
        return self.con.execute(sql, p).fetchall()

    def load_fake_pipeline(self):
        self.con.executescript((FIX / "fake_pipeline.sql").read_text())
        self.con.commit()

    def test_schema_idempotent(self):
        self.assertEqual(run("--db", self.db, "init")[0], 0)

    def test_fts_triggers(self):
        self.load_fake_pipeline()
        self.assertEqual(len(self.q("SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'FAKEBIRD'")), 1)
        self.assertEqual(len(self.q("SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'ZZTEST'")), 3)
        self.con.execute("UPDATE messages SET message_text='ZZNEWWORDCCCCCCCCCCCCCCCCCCCCC' WHERE id=3")
        self.assertEqual(len(self.q("SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'ZZTEST'")), 2)
        self.con.execute("DELETE FROM messages WHERE id=4")
        self.assertEqual(len(self.q("SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'FAKEBIRD'")), 0)
        self.con.execute("INSERT INTO messages_fts(messages_fts) VALUES('integrity-check')")

    def test_fill_from_capture_and_checks(self):
        self.load_fake_pipeline()
        self.assertEqual(self.q("SELECT receiver_id, frequency_khz FROM messages WHERE id=2"), [(2, 8992.0)])
        with self.assertRaises(sqlite3.IntegrityError):
            self.con.execute("INSERT INTO messages(heard_utc, msg_type) VALUES ('2031-01-01 00:00', 'EAM')")
        with self.assertRaises(sqlite3.IntegrityError):
            self.con.execute("INSERT INTO messages(heard_utc, msg_type) VALUES ('2031-01-01T00:00:00Z', 'NOPE')")

    def test_link_simulcast_and_rebroadcast(self):
        self.load_fake_pipeline()
        rc, out = run("--db", self.db, "link")
        self.assertEqual(rc, 0)
        kinds = dict(self.q("SELECT link_kind, count(*) FROM message_links GROUP BY 1"))
        self.assertEqual(kinds, {"simulcast": 1, "rebroadcast": 2})
        self.assertEqual(self.q("SELECT n_messages, n_airings, canonical_text FROM message_clusters"),
                         [(3, 2, "ZZTESTCCCCCCCCCCCCCCCCCCCCCCCC")])
        day = self.q("SELECT eams, eam_broadcasts, eam_receptions FROM v_our_daily_eams")
        self.assertEqual(day, [(2, 3, 4)])     # distinct (default), per broadcast, receptions
        self.assertEqual(self.q("SELECT eams, eam_broadcasts, skykings FROM v_daily_counts"), [(2, 3, 1)])
        run("--db", self.db, "link")  # rerun is stable
        self.assertEqual(self.q("SELECT count(*) FROM message_links")[0][0], 3)

    def test_link_settings_precedence(self):
        self.load_fake_pipeline()
        rc, out = run("--db", self.db, "settings")
        self.assertIn("link.threshold\t0.9", out)
        # settings-table override: with a 0.5 s simulcast window the copies 1 s apart become a rebroadcast
        run("--db", self.db, "settings", "link.simulcast_sec", "0.5")
        run("--db", self.db, "link", "--similarity", "builtin")
        self.assertEqual(dict(self.q("SELECT link_kind, count(*) FROM message_links GROUP BY 1")), {"rebroadcast": 3})
        # CLI flag beats the table
        run("--db", self.db, "link", "--similarity", "builtin", "--simulcast-sec", "120")
        self.assertEqual(dict(self.q("SELECT link_kind, count(*) FROM message_links GROUP BY 1")),
                         {"simulcast": 1, "rebroadcast": 2})
        # a 6-minute rebroadcast window keeps only the simulcast pair (the repeat is 30 min later)
        run("--db", self.db, "link", "--similarity", "builtin", "--simulcast-sec", "120", "--window-hours", "0.1")
        self.assertEqual(dict(self.q("SELECT link_kind, count(*) FROM message_links GROUP BY 1")), {"simulcast": 1})
        self.assertEqual(run("--db", self.db, "settings", "link.similarity", "bogus")[0], 2)

    def test_decoding_similarity_hook_and_fallback(self):
        import types
        self.load_fake_pipeline()
        calls = []

        def fake_text_similarity(a, b):
            calls.append((a, b))
            return 1.0 if a[:6] == b[:6] else 0.0
        pkg = types.ModuleType("decoding"); pkg.__path__ = []
        mod = types.ModuleType("decoding.similarity"); mod.text_similarity = fake_text_similarity
        saved = {k: sys.modules.get(k) for k in ("decoding", "decoding.similarity")}
        try:
            sys.modules["decoding"], sys.modules["decoding.similarity"] = pkg, mod
            rc, out = run("--db", self.db, "link")                       # auto -> decoding
            self.assertTrue(calls)
            self.assertEqual(self.q("SELECT DISTINCT method FROM message_links"), [("decoding:text_similarity",)])
            mod.text_similarity = lambda a, b: 1.5                       # out of range -> error
            with self.assertRaises(ValueError):
                run("--db", self.db, "link")
            sys.modules["decoding.similarity"] = None                    # not importable
            run("--db", self.db, "link")                                 # auto -> builtin
            self.assertEqual(self.q("SELECT DISTINCT method FROM message_links"), [("builtin:wildcard_hamming:v1",)])
            with self.assertRaises(ImportError):
                run("--db", self.db, "link", "--similarity", "decoding")
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_neet_round_trip_exact(self):
        src = FIX / "neet_format_fake.csv"
        self.assertEqual(run("--db", self.db, "import-neet", str(src))[0], 0)
        out = Path(self.tmp.name) / "out.csv"
        self.assertEqual(run("--db", self.db, "export-neet", str(out))[0], 0)
        self.assertEqual(out.read_bytes(), src.read_bytes())
        run("--db", self.db, "import-neet", str(src))           # idempotent
        self.assertEqual(self.q("SELECT count(*) FROM messages")[0][0], 7)
        self.assertEqual(dict(self.q("SELECT msg_type, count(*) FROM messages GROUP BY 1")),
                         {"EAM": 5, "OTHER": 1, "NIL": 1})
        self.assertEqual(self.q("SELECT addressee FROM messages WHERE broadcast_type='directed'"), [("FAKEBIRD 1",)])

    def test_neet_export_of_pipeline_rows(self):
        self.load_fake_pipeline()
        rc, _ = run("--db", self.db, "export-neet", str(Path(self.tmp.name) / "p.csv"))
        rows = (Path(self.tmp.name) / "p.csv").read_text().splitlines()
        self.assertEqual(rows[0], ",".join(hfgcs_db.NEET_COLUMNS))
        self.assertIn("WED,2031.01.01,00:05,TESTCALL,ZZ,ZZTESTCCCCCCCCCCCCCCCCCCCCCCCC,,30,,,,test-rx-a,", rows)
        self.assertTrue(any(",FOR FAKEBIRD 7," in r for r in rows))

    def test_catalog_ingest(self):
        rc, out = run("--db", self.db, "ingest-catalog", str(FIX / "scanner_catalog_fake.jsonl"),
                      "--receiver", "test-rx-a")
        self.assertEqual(rc, 0)
        self.assertEqual(self.q("SELECT start_utc, end_utc FROM captures ORDER BY id LIMIT 1"),
                         [("2031-01-01T00:05:00.500Z", "2031-01-01T00:06:10.500Z")])
        run("--db", self.db, "ingest-catalog", str(FIX / "scanner_catalog_fake.jsonl"), "--receiver", "test-rx-a")
        self.assertEqual(self.q("SELECT count(*) FROM captures")[0][0], 2)

    def test_missileer_import(self):
        f = str(FIX / "missileer_format_fake.json")
        for _ in range(2):
            self.assertEqual(run("--db", self.db, "import-missileer", f)[0], 0)
        self.assertEqual(self.q("SELECT count(*) FROM messages")[0][0], 4)
        self.assertEqual(dict(self.q("SELECT link_kind, count(*) FROM message_links GROUP BY 1")),
                         {"simulcast": 1, "rebroadcast": 1})
        self.assertEqual(self.q("SELECT corrected, review_status, preamble, prefix, heard_at_basis, "
                                "json_array_length(char_times) FROM messages WHERE is_first_reception=1 "
                                "AND msg_type='EAM'"), [(1, "reviewed", "ZZFAKE", "ZZ", "voice_onset", 2)])
        self.assertEqual(self.q("SELECT msg_type, capture_id IS NULL FROM messages WHERE external_id='eam-fake0004'"),
                         [("RADIOCHECK", 1)])
        run("--db", self.db, "link")
        self.assertEqual(self.q("SELECT n_messages, n_airings FROM message_clusters"), [(3, 2)])

    def test_coverage_comparison(self):
        self.load_fake_pipeline()
        run("--db", self.db, "link")
        self.assertEqual(run("--db", self.db, "import-daily-counts", str(FIX / "external_daily_counts_fake.csv"))[0], 0)
        rows = self.q("SELECT date_utc, external_count, our_eams, our_broadcasts, coverage_pct "
                      "FROM v_coverage_comparison ORDER BY 1")
        # coverage is per distinct message: 2 distinct of 5 external = 40 %
        self.assertEqual(rows, [("2031-01-01", 5, 2, 3, 40.0), ("2031-01-02", 2, 0, 0, 0.0)])
        self.assertEqual(self.q("SELECT DISTINCT method FROM external_daily_counts"), [("manual",)])
        with self.assertRaises(sqlite3.IntegrityError):
            self.con.execute("INSERT INTO external_daily_counts(source, date_utc, count, method) "
                             "VALUES ('example_counts', '2031-01-03', 1, 'api')")

    def test_neet_query_translation(self):
        self.load_fake_pipeline()
        rc, out = run("--db", self.db, "query", "3101&C30&PR=ZZ")
        self.assertEqual(len(out.strip().splitlines()), 3)
        rc, out = run("--db", self.db, "query", "--", "--01&Stest-rx-b")
        self.assertEqual(len(out.strip().splitlines()), 1)
        rc, out = run("--db", self.db, "query", '"FAKEBIRD"&R')
        self.assertEqual(len(out.strip().splitlines()), 1)
        with self.assertRaises(ValueError):
            hfgcs_db.neet_query_sql("BOGUS!")


if __name__ == "__main__":
    unittest.main()
