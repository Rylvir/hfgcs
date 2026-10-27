"""HFGCS decoding: turn ASR transcripts into structured EAM / Skyking records.

Stdlib only, so it can run anywhere the scanner runs. See docs/decoding.md.
"""

from .align import align_reads, consensus
from .eam_parser import SCHEMA_VERSION, parse_eam, parse_skyking, parse_transcript
from .metrics import cer, edit_distance
from .rebroadcast import link_rebroadcasts
from .tokens import Token, tokenize

__version__ = "0.3.2"
"""Decoding package version (``decoder_version``'s ``decoding/<version>``).
Bump it on any change that alters decode output."""

__all__ = [
    "SCHEMA_VERSION",
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
