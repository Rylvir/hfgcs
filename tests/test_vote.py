from decoding.vote import apply_vote, medoid, needs_second_pass, vote_bodies


def test_vote_fixes_single_substitution():
    assert vote_bodies(["ABCXEF", "ABCDEF", "ABCDEF"]) == "ABCDEF"


def test_vote_drops_char_most_passes_lack_and_ignores_insertions():
    assert vote_bodies(["ABCDEF", "ABDEF", "ABDEF"]) == "ABDEF"
    assert vote_bodies(["ABCDEF", "ABCDZEF", "ABCDEF"]) == "ABCDEF"


def test_vote_ties_keep_first_pass_and_empty_inputs():
    assert medoid(["ABC", "ABD"]) == 0
    assert vote_bodies(["ABC", "ABD"]) == "ABC"  # two passes: no majority, keep pass 1
    assert vote_bodies(["", "", ""]) == ""
    assert vote_bodies(["", "ABC", "ABC"]) == "ABC"


def test_vote_three_disagreeing_keeps_pivot_char():
    assert vote_bodies(["ABX", "ABY", "ABZ"]) == "ABX"


def _rec(body, agree, flags=()):
    return {"type": "eam", "body": body, "confidence": {"reads_agree_ratio": agree}, "flags": list(flags)}


def test_trigger_rule():
    assert needs_second_pass(_rec("ABCDEF", 0.0, ["single_read"]))
    assert needs_second_pass(_rec("ABCDEF", 0.29))
    assert not needs_second_pass(_rec("ABCDEF", 0.3))
    assert not needs_second_pass(None)  # radio checks / noise never trigger
    assert not needs_second_pass({"type": "unknown", "flags": []})
    assert needs_second_pass(_rec("ABCDEF", 0.9), threshold=1.01)


def test_apply_vote_keeps_first_pass_body_and_flags():
    out = apply_vote(_rec("ABCXEF", 0.0), [_rec("ABCDEF", 0.5), _rec("ABCDEF", 0.5)])
    assert out["body"] == "ABCDEF" and out["body_first_pass"] == "ABCXEF"
    assert out["flags"][-1] == "second_pass_vote" and out["body_length"] == 6
    assert apply_vote(None, [None, None]) is None
