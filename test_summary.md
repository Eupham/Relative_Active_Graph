# CSRRE Test Execution Summary

This document records what the current tests exercise and what they confirm. It is updated alongside the test suite.

---

## 1. Python Induction Pipeline

We streamed a small subset of C4 (500 sentences, English) through the Python induction pipeline.

### What was run

```bash
python lcs/induction/sequential_trainer.py --lang en --max-sents 500
```

The pipeline tokenizes each sentence, extracts surface features, assigns modal categories via bisimulation partition refinement, and accumulates a lexicon via MLE over the resulting MTLG graphs.

### Observed output

```
  Token: The          Lemma: the          Lex Mode: diamond  Lex Cat: Scene
  Token: scientist    Lemma: scientist    Lex Mode: —        Lex Cat: —
  Token: discovered   Lemma: discover     Lex Mode: diamond  Lex Cat: Process
  Token: a            Lemma: a            Lex Mode: diamond  Lex Cat: Scene
  Token: new          Lemma: new          Lex Mode: diamond  Lex Cat: State
  Token: particle     Lemma: particle     Lex Mode: —        Lex Cat: —
  Token: in           Lemma: in           Lex Mode: diamond  Lex Cat: Scene
  Token: the          Lemma: the          Lex Mode: diamond  Lex Cat: Scene
  Token: laboratory   Lemma: laboratory   Lex Mode: —        Lex Cat: —
  Token: .            Lemma: .            Lex Mode: diamond  Lex Cat: Connector

Assigned TRD cluster: en_trd_10
```

Lexicon size after 500 sentences: ~2088 lemmas. TRD clusters found: 16.

### What this confirms

- The induction pipeline runs to completion against a live C4 stream.
- Surface features are extracted and modal categories assigned without an external UD parser.
- The lexicon grows monotonically from the stream.
- TRD clustering produces stable cluster assignments across runs.

### What this does not confirm

- Category labels (Scene, Process, State, …) follow UCCA naming conventions but are assigned by bisimulation partition refinement over surface features, not by a trained UCCA model. The correspondence to UCCA semantics is structural intent, not verified alignment.
- Tokens with `Lex Cat: —` were not assigned a category. This is expected at 500 sentences; coverage improves with more training data.

---

## 2. Rust Linearizer and Type-Filter Unit Test

```
test test_generative_inference_pipeline ... ok
test result: ok. 1 passed; 0 failed
```

### What was run

`core/core/tests/test_generative_inference.rs` — `test_generative_inference_pipeline`

### What the test does

The test manually constructs a `Hypothesis` with a hard-coded predicate (`run(alice)`) and registers the corresponding surface forms directly into the `Linearizer` lexicon. It does not run the induction pipeline or `generate_hypotheses`. It then verifies:

1. `filter_satisfying` accepts the hypothesis against a compatible expected type.
2. `Linearizer::linearize` assembles `"runs Alice"` from the registered entries.

### What this confirms

- The type-filter correctly matches a functor hypothesis against a compatible expected type.
- The linearizer assembles surface strings from registered predicate→surface mappings in root-first order.

### What this does not confirm

- End-to-end generation from unseen input. The hypothesis and lexicon are both provided directly by the test; neither the induction pipeline nor the hypothesis generation search is exercised.

---

## 3. End-to-End IPC Path Test

```
$ python test_e2e_english.py "Alice discovered a particle."
--- Input ---
Alice discovered a particle.

--- Output ---
Alice
```

### What was run

`test_e2e_english.py` — tokenizes the input sentence, converts it to an MTLG graph, serializes it as NDJSON, sends it through the subprocess bridge to the compiled Rust CLI binary, and prints the returned surface token.

### What this confirms

- The tokenizer and MTLG converter run correctly on a plain English sentence.
- The NDJSON payload reaches the CLI and is parsed without error.
- The Rust engine executes a query and returns a JSON response.
- The Python → Rust subprocess IPC path is functional end to end.

### Why the output is a single token

The lexicon is seeded only from the input sentence's surface forms via `register_lexicon` messages immediately before the query. No trained lexicon is loaded at startup. With only the input tokens registered, the engine selects the highest-attribution node in the graph, which in this case is `Alice`.

Multi-token generation requires a persisted, trained lexicon passed to the CLI at startup via `--load-state` or an equivalent mechanism. That is the next milestone for this test.

---

## Current Status

| Capability | Status |
|---|---|
| C4 streaming + lexicon induction | Working |
| TRD clustering | Working |
| Type-filter and linearizer (seeded lexicon) | Working |
| Python → Rust IPC path | Working |
| Multi-token generation from cold start | Not yet demonstrated |
| DRS scope restriction at quantifier boundaries | Not yet implemented |
| UCCA/AMR category assignment via trained parser | Not yet implemented |
| Trained lexicon persistence across sessions | Not yet implemented |

These are the active development targets.
