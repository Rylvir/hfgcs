"""HFGCS decoding: turn ASR transcripts into structured EAM / Skyking records.

Stdlib only, so it can run anywhere the scanner runs. See docs/decoding.md.
"""

from .align import align_reads, consensus
from .eam_parser import parse_eam, parse_skyking, parse_transcript
from .metrics import cer, edit_distance
from .rebroadcast import link_rebroadcasts
from .tokens import Token, tokenize

__all__ = [
    "Token",
    "tokenize",
    "parse_transcript",
    "parse_eam",
    "parse_skyking",
    "align_reads",
    "consensus",
    "cer",
    "edit_distance",
    "link_rebroadcasts",
]

SCHEMA_VERSION = "0.1"
