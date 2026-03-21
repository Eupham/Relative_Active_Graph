"""Universal Dependencies (UD) parser wrapper using Stanza.

Produces UD dependency trees from sentences in any of the 108 mC4 languages.
Language-agnostic annotation scheme: the same UD relation set applies to all languages.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# UD dependency relation types used in MTLG edge assignment.
UD_CORE_DEPS    = {"nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp"}
UD_NONCORE_DEPS = {"obl", "vocative", "expl", "dislocated", "advcl", "advmod",
                   "discourse", "aux", "cop", "mark"}
UD_MODIFIER_DEPS = {"nmod", "appos", "nummod", "acl", "amod", "det", "clf"}
UD_SPECIAL_DEPS  = {"conj", "cc", "fixed", "flat", "compound", "list", "parataxis",
                     "orphan", "goeswith", "reparandum", "root", "dep"}

# Enhanced UD: reentrancy (shared arguments) → □ mode edges.
REENTRANT_RELATIONS = {"nsubj:outer", "obj:outer", "nsubj:xsubj"}


@dataclass
class UdToken:
    id:       int
    text:     str
    lemma:    str
    upos:     str          # Universal POS tag
    xpos:     str          # Language-specific POS
    head:     int          # Head token ID (0 = root)
    deprel:   str          # UD dependency relation
    deps:     str = ""     # Enhanced dependencies (for reentrancy)
    feats:    dict = field(default_factory=dict)

    @property
    def is_root(self) -> bool:
        return self.deprel == "root"

    @property
    def is_reentrant(self) -> bool:
        """Shared argument in enhanced UD → □ mode in MTLG."""
        return any(r in self.deps for r in REENTRANT_RELATIONS)


@dataclass
class UdTree:
    tokens:   list[UdToken]
    language: str
    text:     str

    def token_by_id(self, tid: int) -> Optional[UdToken]:
        return next((t for t in self.tokens if t.id == tid), None)

    def dependents_of(self, head_id: int) -> list[UdToken]:
        return [t for t in self.tokens if t.head == head_id]

    def root_tokens(self) -> list[UdToken]:
        return [t for t in self.tokens if t.is_root]


class UdParser:
    """Wraps Stanza for multilingual UD parsing."""

    # Cache of loaded Stanza pipelines (one per language).
    _pipelines: dict = {}

    def __init__(self, language: str = "en"):
        self.language = language
        self._pipeline = self._load_pipeline(language)

    @classmethod
    def _load_pipeline(cls, language: str):
        if language in cls._pipelines:
            return cls._pipelines[language]
        try:
            import stanza
            # Download model if not present.
            stanza.download(language, verbose=False)
            pipeline = stanza.Pipeline(
                language,
                processors="tokenize,mwt,pos,lemma,depparse",
                verbose=False,
                use_gpu=False,
            )
            cls._pipelines[language] = pipeline
            logger.info("Loaded Stanza pipeline for language=%s", language)
            return pipeline
        except Exception as exc:
            logger.error("Failed to load Stanza pipeline for %s: %s", language, exc)
            raise

    def parse(self, sentence: str) -> UdTree:
        """Parse a sentence and return a UdTree."""
        doc = self._pipeline(sentence)
        tokens = []
        for sent in doc.sentences:
            for word in sent.words:
                tokens.append(UdToken(
                    id=word.id,
                    text=word.text,
                    lemma=word.lemma or word.text,
                    upos=word.upos or "X",
                    xpos=word.xpos or "_",
                    head=word.head,
                    deprel=word.deprel or "dep",
                    deps=word.deps or "",
                    feats=dict(f.split("=") for f in (word.feats or "").split("|") if "=" in f),
                ))
        return UdTree(tokens=tokens, language=self.language, text=sentence)

    def parse_batch(self, sentences: list[str]) -> list[UdTree]:
        return [self.parse(s) for s in sentences]


if __name__ == "__main__":
    import json
    parser = UdParser("en")
    tree = parser.parse("Alice runs quickly.")
    for tok in tree.tokens:
        print(json.dumps({
            "id": tok.id, "text": tok.text, "upos": tok.upos,
            "head": tok.head, "deprel": tok.deprel,
        }))
