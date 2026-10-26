-- CLEARLY FAKE sample rows (year 2031, 'ZZ'/'TEST' strings, example.invalid URLs).
INSERT INTO receivers(code, name, kind, url, latitude, longitude, notes) VALUES
 ('test-rx-a', 'FAKE receiver A', 'local_sdr', NULL, 45.0, -122.0, 'fake'),
 ('test-rx-b', 'FAKE receiver B', 'kiwisdr', 'http://example.invalid:8073', 60.0, 10.0, 'fake');
INSERT INTO captures(receiver_id, frequency_khz, start_utc, end_utc, duration_sec, snr_db, noise_floor_db, audio_path, classification)
SELECT r.id, v.f, v.s, v.e, v.d, v.snr, v.nf, v.path, 'voice' FROM (
  SELECT 'test-rx-a' AS rx, 11175 AS f, '2031-01-01T00:05:00Z' AS s, '2031-01-01T00:07:30Z' AS e, 150 AS d, 14.2 AS snr, -72.0 AS nf, 'recordings/FAKE_a_11175.flac' AS path
  UNION ALL SELECT 'test-rx-b',  8992, '2031-01-01T00:05:01Z', '2031-01-01T00:07:31Z', 150,  9.1, -75.5, 'recordings/FAKE_b_8992.flac'
  UNION ALL SELECT 'test-rx-a', 11175, '2031-01-01T00:35:00Z', '2031-01-01T00:37:00Z', 120, 11.0, -71.0, 'recordings/FAKE_a_11175_2.flac'
  UNION ALL SELECT 'test-rx-a',  4724, '2031-01-01T03:10:00Z', '2031-01-01T03:11:00Z',  60,  8.5, -80.0, 'recordings/FAKE_a_4724.flac'
) v JOIN receivers r ON r.code = v.rx;
INSERT INTO messages(capture_id, heard_utc, heard_at_basis, offset_sec, msg_type, broadcast_type, callsign, prefix, preamble, message_text, length, announced_length, addressee, confidence, transcript, review_status) VALUES
 ((SELECT id FROM captures WHERE audio_path='recordings/FAKE_a_11175.flac'), '2031-01-01T00:05:20Z', 'voice_onset', 20, 'EAM', 'broadcast', 'TESTCALL', 'ZZ', 'ZZTEST', 'ZZTESTCCCCCCCCCCCCCCCCCCCCCCCC', 30, 30, NULL, 0.97, 'zulu zulu tango echo ...', 'auto'),
 ((SELECT id FROM captures WHERE audio_path='recordings/FAKE_b_8992.flac'), '2031-01-01T00:05:21Z', 'voice_onset', 20, 'EAM', 'broadcast', 'TESTCALL', 'ZZ', 'ZZTEST', 'ZZTESTCCCCCCCCCC.CCCCCCCCCCCCC', 30, 30, NULL, 0.81, NULL, 'auto'),
 ((SELECT id FROM captures WHERE audio_path='recordings/FAKE_a_11175_2.flac'), '2031-01-01T00:35:15Z', 'voice_onset', 15, 'EAM', 'broadcast', 'TESTCALL', 'ZZ', 'ZZTEST', 'ZZTESTCCCCCCCCCCCCCCCCCCCCCCCC', 30, 30, NULL, 0.95, NULL, 'auto'),
 ((SELECT id FROM captures WHERE audio_path='recordings/FAKE_a_11175.flac'), '2031-01-01T00:06:40Z', 'voice_onset', 100, 'EAM', 'directed', 'TESTCALL', 'QX', 'QXFAKE', 'QXFAKEDDDDDDDDDDDDDDDDDDDDDDDD', 30, 30, 'FAKEBIRD 7', 0.90, NULL, 'auto');
INSERT INTO messages(capture_id, heard_utc, msg_type, broadcast_type, callsign, message_text, skyking_codeword, skyking_time, skyking_auth, confidence) VALUES
 ((SELECT id FROM captures WHERE audio_path='recordings/FAKE_a_4724.flac'), '2031-01-01T03:10:10Z', 'SKYKING', 'broadcast', 'SKYKING', 'FAKEWORD TIME 12 AUTH ZZ', 'FAKEWORD', '12', 'ZZ', 0.88);
INSERT INTO external_sources(code, name, license, notes) VALUES ('example_counts', 'FAKE example counter', NULL, 'tests only');
