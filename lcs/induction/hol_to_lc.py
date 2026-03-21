"""HOL → LC → MLTT bridge (Curry-Howard correspondence).

ccg2lambda produces Higher-Order Logic (HOL) formulas.
This module bridges HOL output to simply typed lambda calculus (LC),
and from LC to Martin-Löf Type Theory (MLTT) for Z3 validation.

Key mappings (Curry-Howard):
  HOL existential ∃x.P(x)   → Σ-type  (a, P(a))
  HOL universal   ∀x.P(x)   → Π-type  λx.P(x)
  HOL conjunction P ∧ Q     → product type P × Q
  HOL implication P → Q     → function type P → Q
  HOL propositional equality → identity type Id(a, b)

Intensional constructions (belief reports, modals, conditionals):
  belief(x, P)  → Π(w: World). P(w)  (intensional type as function over worlds)
  modal(□P)     → Π(w: World). P(w)
  modal(◇P)     → Σ(w: World, P(w))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

# ─── HOL AST ─────────────────────────────────────────────────────────────────

@dataclass
class Var:   name: str
@dataclass
class Const: name: str; sort: str = "e"  # entity sort
@dataclass
class App:   func: "HolTerm"; arg: "HolTerm"
@dataclass
class Abs:   var: str; sort: str; body: "HolTerm"
@dataclass
class Exists:var: str; sort: str; body: "HolTerm"
@dataclass
class Forall:var: str; sort: str; body: "HolTerm"
@dataclass
class And:   left: "HolTerm"; right: "HolTerm"
@dataclass
class Impl:  ante: "HolTerm"; cons: "HolTerm"
@dataclass
class Belief:agent: str; prop: "HolTerm"   # intensional
@dataclass
class Modal: operator: str; prop: "HolTerm" # □ or ◇

HolTerm = Union[Var, Const, App, Abs, Exists, Forall, And, Impl, Belief, Modal]

# ─── LC AST ──────────────────────────────────────────────────────────────────

@dataclass
class LcVar:    name: str
@dataclass
class LcConst:  name: str
@dataclass
class LcApp:    func: "LcTerm"; arg: "LcTerm"
@dataclass
class LcAbs:    var: str; body: "LcTerm"
@dataclass
class LcPair:   fst: "LcTerm"; snd: "LcTerm"
@dataclass
class LcPi:     var: str; body: "LcTerm"    # Π-type (encoded as LC)
@dataclass
class LcSigma:  var: str; body: "LcTerm"    # Σ-type (encoded as LC)
@dataclass
class LcWorld:  term: "LcTerm"              # world-parameterized (intensional)

LcTerm = Union[LcVar, LcConst, LcApp, LcAbs, LcPair, LcPi, LcSigma, LcWorld]

# ─── HOL → LC translation ─────────────────────────────────────────────────────

def hol_to_lc(term: HolTerm) -> LcTerm:
    """Translate a HOL term to simply typed LC via Curry-Howard."""
    if isinstance(term, Var):
        return LcVar(term.name)
    if isinstance(term, Const):
        return LcConst(term.name)
    if isinstance(term, App):
        return LcApp(hol_to_lc(term.func), hol_to_lc(term.arg))
    if isinstance(term, Abs):
        return LcAbs(term.var, hol_to_lc(term.body))
    if isinstance(term, Exists):
        # ∃x:A.P  →  Σ(x:A, P)  (dependent pair)
        return LcSigma(term.var, hol_to_lc(term.body))
    if isinstance(term, Forall):
        # ∀x:A.P  →  Π(x:A, P)  (dependent function)
        return LcPi(term.var, hol_to_lc(term.body))
    if isinstance(term, And):
        # P ∧ Q  →  P × Q
        return LcPair(hol_to_lc(term.left), hol_to_lc(term.right))
    if isinstance(term, Impl):
        # P → Q  →  λ_: P. Q  (function type)
        return LcAbs("_", hol_to_lc(term.cons))
    if isinstance(term, Belief):
        # belief(x, P)  →  Π(w: World). P(w)
        return LcWorld(LcPi("w", hol_to_lc(term.prop)))
    if isinstance(term, Modal):
        if term.operator == "□":
            # □P  →  Π(w: World). P(w)
            return LcWorld(LcPi("w", hol_to_lc(term.prop)))
        else:  # ◇
            # ◇P  →  Σ(w: World, P(w))
            return LcWorld(LcSigma("w", hol_to_lc(term.prop)))
    raise ValueError(f"Unhandled HOL term: {type(term).__name__}")


# ─── LC display / serialization ───────────────────────────────────────────────

def lc_to_str(term: LcTerm, parens: bool = False) -> str:
    """Render an LC term as a string (for inspection and passing to Z3)."""
    if isinstance(term, LcVar):
        return term.name
    if isinstance(term, LcConst):
        return term.name
    if isinstance(term, LcApp):
        s = f"{lc_to_str(term.func, True)} {lc_to_str(term.arg, True)}"
        return f"({s})" if parens else s
    if isinstance(term, LcAbs):
        s = f"λ{term.var}.{lc_to_str(term.body)}"
        return f"({s})" if parens else s
    if isinstance(term, LcPair):
        return f"({lc_to_str(term.fst)}, {lc_to_str(term.snd)})"
    if isinstance(term, LcPi):
        return f"Π{term.var}.{lc_to_str(term.body)}"
    if isinstance(term, LcSigma):
        return f"Σ{term.var}.{lc_to_str(term.body)}"
    if isinstance(term, LcWorld):
        return f"World[{lc_to_str(term.term)}]"
    return "?"


if __name__ == "__main__":
    # Example: ∃x. run(x)  →  Σ(x, run(x))
    hol = Exists("x", "e", App(Const("run"), Var("x")))
    lc  = hol_to_lc(hol)
    print(lc_to_str(lc))  # Σx.(run x)

    # Example: □believe(a, ∀x. P(x))
    hol2 = Modal("□", Belief("alice", Forall("x", "e", App(Const("P"), Var("x")))))
    lc2  = hol_to_lc(hol2)
    print(lc_to_str(lc2))
