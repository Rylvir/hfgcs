from decoding.preamble import fuse_preamble, preamble_candidates
from decoding.vote import vote_on_pivot


def _rec(pre_reads, read1, read2, body=None, preamble=None, flags=()):
    return {
        "type": "eam",
        "preamble": preamble,
        "preamble_reads": pre_reads,
        "read1": read1,
        "read2": read2,
        "body": body if body is not None else read1,
        "flags": list(flags),
    }


def test_candidates_cut_junk_prefix_and_use_read_starts():
    full, partial = preamble_candidates(_rec(["344QHSI7J", "QHSI7", "QHSI7J"], "QHSI7JAB", "QHSXXJAB"))
    assert full == ["QHSI7J", "QHSI7J", "QHSI7J", "QHSXXJ"] and partial == ["QHSI7"]


def test_mid_message_start_ignores_read_starts():
    full, _ = preamble_candidates(_rec(["ABCDEF"], "XYZXYZ", "", flags=["mid_message_start"]))
    assert full == ["ABCDEF"]


def test_fuse_votes_over_all_hearings_and_patches_body():
    rec = _rec(["QHCBK", "QHCPBK"], "QQHCPBK6AB", "HCPBK6AB", preamble="QHCBK")
    out = fuse_preamble(rec)
    assert out["preamble"] == "QHCPBK" and "preamble_fused" in out["flags"]
    assert out["body"] == "QHCPBK6AB" and "body_start_from_preamble" in out["flags"]
    assert rec["body"] == "QQHCPBK6AB"  # input not mutated
    assert fuse_preamble(rec, patch_body=False)["body"] == "QQHCPBK6AB"


def test_fuse_leaves_far_body_start_and_non_eam_alone():
    out = fuse_preamble(_rec(["ABCDEF", "ABCDEF"], "ZZZZZZ12", "", body="ZZZZZZ12"))
    assert out["preamble"] == "ABCDEF" and out["body"] == "ZZZZZZ12"
    assert fuse_preamble(None) is None
    assert fuse_preamble({"type": "skyking"}) == {"type": "skyking"}
    lone = _rec(["ABCDEF"], "", "")
    assert fuse_preamble(lone) is lone  # one hearing: nothing to vote


def test_vote_on_pivot_counts_pivot_once():
    assert vote_on_pivot("ABX", ["ABY", "ABY"]) == "ABY"
    assert vote_on_pivot("ABX", ["ABY"]) == "ABX"
