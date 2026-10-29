"""Stdlib tests for schema.sql and scripts/hfgcs_db.py. All data is fake. Run:
    python3 -m unittest discover -s tests -v
"""
import contextlib
import importlib
import importlib.util
import io
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "scripts"))
hfgcs_db = importlib.import_module("hfgcs_db")


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

    def link_kinds(self):
        return dict(self.q("SELECT link_kind, count(*) FROM message_links GROUP BY 1"))

    def link_methods(self):
        return self.q("SELECT DISTINCT method FROM message_links")

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
            self.con.execute("INSERT INTO messages(heard_utc, msg_type) "
                             "VALUES ('2031-01-01T00:00:00Z', 'NOPE')")

    def test_link_simulcast_and_rebroadcast(self):
        self.load_fake_pipeline()
        rc, out = run("--db", self.db, "link")
        self.assertEqual(rc, 0)
        kinds = self.link_kinds()
        self.assertEqual(kinds, {"simulcast": 1, "rebroadcast": 2})
        self.assertEqual(self.q("SELECT n_messages, n_airings, canonical_text FROM message_clusters"),
                         [(3, 2, "ZZTESTCCCCCCCCCCCCCCCCCCCCCCCC")])
        day = self.q("SELECT eams, eam_broadcasts, eam_receptions FROM v_our_daily_eams")
        self.assertEqual(day, [(2, 3, 4)])     # distinct (default), per broadcast, receptions
        self.assertEqual(self.q("SELECT eams, eam_broadcasts, skykings FROM v_daily_counts"), [(2, 3, 1)])
        run("--db", self.db, "link")  # rerun is stable
        self.assertEqual(self.q("SELECT count(*) FROM message_links")[0][0], 3)

    def test_link_rows_loaded_out_of_time_order(self):
        # a backfill can give a later-heard row the smaller id; the link must still be stored
        self.con.execute("""INSERT INTO messages(heard_utc, msg_type, prefix, message_text) VALUES
                            ('2031-01-01T02:00:00Z', 'EAM', 'ZZ', 'ZZTESTCCCCCCCCCCCCCCCCCCCCCCCC'),
                            ('2031-01-01T01:00:00Z', 'EAM', 'ZZ', 'ZZTESTCCCCCCCCCCCCCCCCCCCCCCCC')""")
        self.con.commit()
        run("--db", self.db, "link", "--similarity", "builtin")
        self.assertEqual(self.q("SELECT message_a, message_b, link_kind, delta_sec FROM message_links"),
                         [(1, 2, "rebroadcast", 3600.0)])
        self.assertEqual(self.q("SELECT n_messages FROM message_clusters"), [(2,)])

    def test_link_settings_precedence(self):
        self.load_fake_pipeline()
        rc, out = run("--db", self.db, "settings")
        self.assertIn("link.threshold\t0.9", out)
        # settings-table override: with a 0.5 s simulcast window the copies 1 s apart become a rebroadcast
        run("--db", self.db, "settings", "link.simulcast_sec", "0.5")
        run("--db", self.db, "link", "--similarity", "builtin")
        self.assertEqual(self.link_kinds(), {"rebroadcast": 3})
        # CLI flag beats the table
        run("--db", self.db, "link", "--similarity", "builtin", "--simulcast-sec", "120")
        self.assertEqual(self.link_kinds(),
                         {"simulcast": 1, "rebroadcast": 2})
        # a 6-minute rebroadcast window keeps only the simulcast pair (the repeat is 30 min later)
        run("--db", self.db, "link", "--similarity", "builtin",
            "--simulcast-sec", "120", "--window-hours", "0.1")
        self.assertEqual(self.link_kinds(), {"simulcast": 1})
        self.assertEqual(run("--db", self.db, "settings", "link.similarity", "bogus")[0], 2)

    def test_decoding_similarity_hook_and_fallback(self):
        import types
        self.load_fake_pipeline()
        calls = []

        def fake_text_similarity(a, b):
            calls.append((a, b))
            return 1.0 if a[:6] == b[:6] else 0.0
        pkg = types.ModuleType("decoding")
        pkg.__path__ = []
        mod = types.ModuleType("decoding.similarity")
        mod.text_similarity = fake_text_similarity
        saved = {k: sys.modules.get(k) for k in ("decoding", "decoding.similarity")}
        try:
            sys.modules["decoding"], sys.modules["decoding.similarity"] = pkg, mod
            rc, out = run("--db", self.db, "link")                       # auto -> decoding
            self.assertTrue(calls)
            self.assertEqual(self.link_methods(), [("decoding:text_similarity",)])
            mod.text_similarity = lambda a, b: 1.5                       # out of range -> error
            with self.assertRaises(ValueError):
                run("--db", self.db, "link")
            sys.modules["decoding.similarity"] = None                    # not importable
            run("--db", self.db, "link")                                 # auto -> builtin
            self.assertEqual(self.link_methods(), [("builtin:wildcard_hamming:v1",)])
            with self.assertRaises(ImportError):
                run("--db", self.db, "link", "--similarity", "decoding")
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def _add_short_copy(self):
        # a copy with one character dropped (29 chars), 1 h after the first airing
        self.con.execute("""INSERT INTO messages(heard_utc, msg_type, callsign, prefix, message_text, length)
                            VALUES ('2031-01-01T01:05:00Z', 'EAM', 'TESTCALL', 'ZZ',
                                    'ZZTESTCCCCCCCCCCCCCCCCCCCCCCC', 29)""")
        self.con.commit()

    def test_unequal_lengths_only_with_decoding_similarity(self):
        import types
        self.load_fake_pipeline()
        self._add_short_copy()
        run("--db", self.db, "link", "--similarity", "builtin",      # builtin: equal lengths only
            "--partial-min-len", "0")                                # (partial links off)
        self.assertEqual(self.q("SELECT n_messages FROM message_clusters"), [(3,)])
        pkg = types.ModuleType("decoding")
        pkg.__path__ = []
        mod = types.ModuleType("decoding.similarity")
        mod.text_similarity = lambda a, b: 1.0 - abs(len(a) - len(b)) / max(len(a), len(b)) \
            if a[:6] == b[:6] else 0.0
        saved = {k: sys.modules.get(k) for k in ("decoding", "decoding.similarity")}
        try:
            sys.modules["decoding"], sys.modules["decoding.similarity"] = pkg, mod
            run("--db", self.db, "link", "--similarity", "decoding")
            self.assertEqual(self.q("SELECT n_messages, length, canonical_text FROM message_clusters"),
                             [(4, 30, "ZZTESTCCCCCCCCCCCCCCCCCCCCCCCC")])   # modal-length consensus
            run("--db", self.db, "link", "--similarity", "decoding", "--max-len-diff", "0",
                "--partial-min-len", "0")
            self.assertEqual(self.q("SELECT n_messages FROM message_clusters"), [(3,)])
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_partial_similarity(self):
        ps = hfgcs_db.partial_similarity
        full = "ZZTESTABCDEFGHIJKLMNOPQRSTUVWX"
        self.assertEqual(ps(full[:14], full), 1.0)               # cut at the end
        self.assertEqual(ps(full, full[10:]), 1.0)               # started mid-message, either order
        self.assertAlmostEqual(ps("ZZTESTABCDEFGHIZ", full), 15 / 16)
        self.assertIsNone(ps(full, full))                        # equal lengths: not a partial
        self.assertIsNone(ps(full[:11], full))                   # shorter than min_len
        self.assertIsNone(ps("ZZTEST........", full))            # too few known characters
        self.assertLess(ps("QQQQQQQQQQQQQQ", full), 0.1)
        # stop_below only drops offsets that cannot reach it; the score of one that can is unchanged
        self.assertIsNone(ps("QQQQQQQQQQQQQQ", full, stop_below=0.9))
        self.assertAlmostEqual(ps("ZZTESTABCDEFGHIZ", full, stop_below=0.9), 15 / 16)
        self.assertIsNone(ps("ZZTESTABCDEFGHZZ", full, stop_below=0.9))   # 14/16 = 0.875
        # the str.find shortcut and the early exit give the same answer as checking every offset
        import random

        def reference(a, b, stop_below=None, min_len=12, min_known=0.5):
            short, full_ = sorted((a, b), key=len)
            n = len(short)
            if n == len(full_) or n < min_len:
                return None
            max_miss = n if stop_below is None else (1.0 - stop_below) * n
            scores = []
            for off in range(len(full_) - n + 1):
                pairs = [(x, y) for x, y in zip(short, full_[off:off + n], strict=True)
                         if x not in ".?" and y not in ".?"]
                miss = sum(x != y for x, y in pairs)
                if len(pairs) >= max(min_len, min_known * n) and miss <= max_miss:
                    scores.append(1 - miss / len(pairs))
            return max(scores, default=None)

        rnd = random.Random(3)
        for _ in range(2000):
            alpha = rnd.choice(["ABC", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", "AB.", "ABCD?"])
            long_ = "".join(rnd.choice(alpha) for _ in range(rnd.randint(12, 45)))
            off = rnd.randint(0, 10)
            frag = list(long_[off:off + rnd.randint(10, 35)])
            for _ in range(rnd.randint(0, 5)):
                if frag:
                    frag[rnd.randrange(len(frag))] = rnd.choice(alpha)
            frag = "".join(frag)
            for stop in (None, 0.9, 0.5):
                self.assertEqual(ps(frag, long_, stop_below=stop), reference(frag, long_, stop),
                                 (frag, long_, stop))

    def test_link_partial_copies(self):
        self.load_fake_pipeline()
        # a fragment cut off after 16 characters, heard alongside the first airing on a
        # third receiver, and a mid-message fragment 2 h later; its "prefix" is not the real one
        self.con.execute("""INSERT INTO messages(heard_utc, msg_type, callsign, prefix, message_text, partial)
                            VALUES ('2031-01-01T00:05:40Z', 'EAM', 'TESTCALL', 'ZZ',
                                    'ZZTESTCCCCCCCCCC', 1),
                                   ('2031-01-01T02:00:00Z', 'EAM', NULL, 'CC',
                                    'CCCCCCCCCCCCCCCCCC', 1),
                                   ('2031-01-01T02:10:00Z', 'EAM', NULL, 'QQ',
                                    'QQQQQQQQQQQQQQQQQQ', 1)""")
        self.con.commit()
        rc, out = run("--db", self.db, "link", "--similarity", "builtin")
        self.assertEqual(rc, 0)
        self.assertIn("'partial'", out)
        kinds = self.link_kinds()
        self.assertEqual(kinds["simulcast"], 1)
        self.assertEqual(kinds["rebroadcast"], 2)
        self.assertGreaterEqual(kinds["partial"], 2)
        self.assertEqual(self.q("SELECT DISTINCT method FROM message_links WHERE link_kind='partial'"),
                         [("builtin:partial_align:v1",)])
        # one cluster; the vote ignores the fragments; the fragment heard with the first
        # airing is the same emission, the later one is a third airing
        self.assertEqual(self.q("SELECT n_messages, n_airings, prefix, length, canonical_text "
                                "FROM message_clusters"),
                         [(5, 3, "ZZ", 30, "ZZTESTCCCCCCCCCCCCCCCCCCCCCCCC")])
        self.assertEqual(self.q("SELECT cluster_id FROM messages WHERE message_text LIKE 'QQ%'"), [(None,)])
        n = self.q("SELECT count(*) FROM message_links")[0][0]
        run("--db", self.db, "link", "--similarity", "builtin")   # rerun is stable
        self.assertEqual(self.q("SELECT count(*) FROM message_links")[0][0], n)
        run("--db", self.db, "settings", "link.partial_min_len", "0")
        run("--db", self.db, "link", "--similarity", "builtin")
        self.assertNotIn("partial", self.link_kinds())
        self.assertEqual(self.q("SELECT n_messages FROM message_clusters"), [(3,)])

    @unittest.skipUnless(importlib.util.find_spec("decoding") is not None
                         and importlib.util.find_spec("decoding.similarity") is not None,
                         "Decoding's decoding.similarity is not on the path")
    def test_real_decoding_similarity(self):
        from decoding.similarity import text_similarity
        self.assertEqual(text_similarity("ZZTEST.CCC", "ZZTESTCCCC"), 1.0)
        with self.assertRaises(TypeError):
            text_similarity(None, "ZZTESTCCCC")
        self.load_fake_pipeline()
        self._add_short_copy()
        rc, out = run("--db", self.db, "link", "--similarity", "decoding")
        self.assertIn("decoding:text_similarity", out)
        self.assertEqual(self.link_kinds(),
                         {"simulcast": 1, "rebroadcast": 5})
        self.assertEqual(self.q("SELECT n_messages, n_airings FROM message_clusters"), [(4, 3)])
        # on equal lengths the outcome matches the built-in
        run("--db", self.db, "link", "--similarity", "decoding", "--max-len-diff", "0")
        dec = self.q("SELECT message_a, message_b, link_kind FROM message_links ORDER BY 1, 2")
        run("--db", self.db, "link", "--similarity", "builtin")
        builtin = self.q("SELECT message_a, message_b, link_kind FROM message_links ORDER BY 1, 2")
        self.assertEqual(dec, builtin)

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
        self.assertEqual(self.q("SELECT addressee FROM messages WHERE broadcast_type='directed'"),
                         [("FAKEBIRD 1",)])

    def test_neet_export_of_pipeline_rows(self):
        self.load_fake_pipeline()
        rc, _ = run("--db", self.db, "export-neet", str(Path(self.tmp.name) / "p.csv"))
        rows = (Path(self.tmp.name) / "p.csv").read_text().splitlines()
        self.assertEqual(rows[0], ",".join(hfgcs_db.NEET_COLUMNS))
        self.assertIn("WED,2031.01.01,00:05,TESTCALL,ZZ,ZZTESTCCCCCCCCCCCCCCCCCCCCCCCC,,30,,,,test-rx-a,",
                      rows)
        self.assertTrue(any(",FOR FAKEBIRD 7," in r for r in rows))

    def test_catalog_ingest(self):
        rc, out = run("--db", self.db, "ingest-catalog", str(FIX / "scanner_catalog_fake.jsonl"),
                      "--receiver", "test-rx-a")
        self.assertEqual(rc, 0)
        self.assertEqual(self.q("SELECT start_utc, end_utc FROM captures ORDER BY id LIMIT 1"),
                         [("2031-01-01T00:05:00.500Z", "2031-01-01T00:06:10.500Z")])
        run("--db", self.db, "ingest-catalog", str(FIX / "scanner_catalog_fake.jsonl"),
            "--receiver", "test-rx-a")
        self.assertEqual(self.q("SELECT count(*) FROM captures")[0][0], 2)

    def test_missileer_import(self):
        f = str(FIX / "missileer_format_fake.json")
        for _ in range(2):
            self.assertEqual(run("--db", self.db, "import-missileer", f)[0], 0)
        self.assertEqual(self.q("SELECT count(*) FROM messages")[0][0], 4)
        self.assertEqual(self.link_kinds(),
                         {"simulcast": 1, "rebroadcast": 1})
        self.assertEqual(self.q("SELECT corrected, review_status, preamble, prefix, heard_at_basis, "
                                "json_array_length(char_times) FROM messages WHERE is_first_reception=1 "
                                "AND msg_type='EAM'"), [(1, "reviewed", "ZZFAKE", "ZZ", "voice_onset", 2)])
        self.assertEqual(self.q("SELECT msg_type, capture_id IS NULL FROM messages "
                                "WHERE external_id='eam-fake0004'"),
                         [("RADIOCHECK", 1)])
        run("--db", self.db, "link")
        self.assertEqual(self.q("SELECT n_messages, n_airings FROM message_clusters"), [(3, 2)])

    def test_coverage_comparison(self):
        self.load_fake_pipeline()
        run("--db", self.db, "link")
        counts_csv = str(FIX / "external_daily_counts_fake.csv")
        self.assertEqual(run("--db", self.db, "import-daily-counts", counts_csv)[0], 0)
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
