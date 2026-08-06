# Launch post draft (for you to post — I have no posting access to Reddit/Discord)

Fill in the [GitHub URL] and [PyPI URL] once you've pushed/published. All
numbers below are from an actual run in this repo (`benchmarks/RESULTS.md`),
not illustrative — re-run `python benchmarks/run_benchmark.py` yourself to
verify before posting.

---

**Title:** tok-adapt — open-source CLI for adapting HF tokenizers to new
languages (78% token reduction on Hindi/Telugu vs. Qwen2.5-7B, measured)

I built a small library for the recurring problem of adapting a base LLM's
tokenizer to a new language or domain without retraining from scratch:
expand vocabulary, smart-initialize the new embeddings from subword means
(not random noise), prune unused vocabulary, and measure the payoff.

**Real, reproducible benchmark** (script + full methodology in the repo):
training a domain BPE sub-tokenizer on real Hindi + Telugu Wikipedia text and
merging it into `Qwen/Qwen2.5-7B`'s tokenizer, then measuring on held-out
articles not seen during training:

- Qwen2.5-7B: 381,277 → 82,774 tokens (**78.3% reduction**)
- gpt2 (sanity check, English-centric baseline): 680,347 → 83,576 tokens
  (**87.7% reduction**)
- Projected embedding-memory savings from pruning Qwen2.5-7B's vocab down to
  only the tokens this domain uses: **~2 GiB freed** (99.2% of the embedding
  matrix), computed from real model config dimensions.

Also worth mentioning since it's the kind of thing that matters for trust:
building this benchmark caught a real bug in the library's own expansion
logic (byte-level vocab remapping meant newly "added" tokens could silently
fail to match real non-ASCII text) — full writeup of the bug and the fix is
in `benchmarks/README.md`, along with the regression tests that now guard
against it.

- Repo: [GitHub URL]
- PyPI: `pip install tok-adapt` [PyPI URL]
- CLI: `tok-adapt expand|align-embeddings|prune|benchmark`

Feedback / issues welcome, especially from anyone who's hit tokenizer
fertility problems adapting a model to an underrepresented language.

---

## Notes for you before posting

- I have not verified this library against Llama-3-8B's tokenizer end-to-end
  (only confirmed it's accessible in this environment); consider running
  `python benchmarks/run_benchmark.py` with `BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B"`
  if you want that number specifically before claiming it publicly.
- r/MachineLearning has stricter self-promotion rules than r/LocalLLaMA —
  check current subreddit rules before posting; a "Research"-flaired framing
  emphasizing the benchmark methodology (not just "I built X") tends to fare
  better there.
- The embedding-memory figure is explicitly labeled as analytical/projected
  in the writeup above — keep that qualifier in any post, don't round it up
  to "measured VRAM savings."
