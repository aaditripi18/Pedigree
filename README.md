# Pedigree

**Do models on the Hugging Face Hub come from the model they say they do?** Pedigree checks a
model's declared `base_model` against its weights, and audits a frozen random sample of 600 Hub
models to measure how often declared lineage is missing, unverifiable or contradicted.
Project site: **https://aaditripi18.github.io/Pedigree/**

Pedigree does not claim a new lineage detector. Weight-based lineage methods already exist and
are strong (AWM, ICLR 2026; HuRef, NeurIPS 2024; MoTHer, ICLR 2025; see [PLAN.md](PLAN.md)).
Pedigree uses that idea as an instrument, measures the instrument on ground truth first, and then
measures the Hub. Every hypothesis was written down, with its decision rule, before its data
existed ([PLAN.md](PLAN.md), with dated amendments); every verdict below is computed by
[`scripts/analyse.py`](scripts/analyse.py) from the append-only ledger
[`results/ledger.jsonl`](results/ledger.jsonl), and none is reworded here.

## What the Hub audit found (E2: 600 models, snapshot 2026-11-04)

The sample is 200 models drawn uniformly from each of three download tiers of the 100,000 most
downloaded models (ranks 1-1k, 1k-10k, 10k-100k; seed 20261105). Every model got exactly one
category, with its reason. The second column is the same 600 models scored again after the file
reader fixes described [below](#every-caveat-measured) (E2r, E2g, E2p, E2f); no verdict changed.

| Category | Models (audit) | After reader fixes |
|---|--:|--:|
| Declares a base, and the weights confirm it | 214 | 246 |
| Declares a base the weights contradict | 3 | 3 |
| Declares a base that cannot be compared by weights | 84 | 52 |
| Declares no base; an earlier related model was found | 105 | 106 |
| Declares no base; no related earlier model found | 136 | 135 |
| Weights not readable (ONNX-only, gated without a public copy, no weights) | 46 | 46 |
| An adapter only (LoRA weights, checked against its base's shapes) | 12 | 12 |

- **Undeclared lineage (H2, not supported).** Of the 364 models whose weights show a related
  earlier model, 105 (28.8%, Wilson 95% interval 24.4-33.7%) declare no base. The pre-registered
  bar was 30%. Weighted by tier size the share is 19.8%: undeclared lineage is commoner among the
  most downloaded models (38 of 101 in the top 1k) than in the long tail (26 of 143).
- **Contradicted declarations (H3, not supported).** Of 227 declared bases that could be compared,
  3 (1.3%, interval 0.5-3.8%) are contradicted: the weights show no lineage at all (median
  cosine below 0.004 against a threshold of 0.0255). The bar was 5%. When a model card names a
  base and the weights can be compared, the claim is almost always true.
- **Unverifiable declarations.** 84 models declare a base that could not be compared by weights
  (94 model-base pairs): 67 pairs share fewer than 8 aligned tensors (65 share none), 23 bases
  are unreadable (11 private, 4 gated without a public copy, 5 legacy pickle files, 3 other) and
  4 comparisons failed.
  Pedigree's second instrument, behaviour, was pre-registered for exactly these. It applied to
  none of the 94 declared pairs involved: 66 have a model above 2B parameters and 28 share no
  output space (14 repackaged files, mostly GGUF, that the standard loader cannot open; 11
  embedding, classification, draft, speech or multimodal models; one without a loadable weight
  file; one gated; one needing custom code). So for these models the Hub's
  declaration was unverifiable by either instrument when the audit ran, and E2 reports them as
  such. Two reader fixes since then (below) make 29 of the 84 comparable by weights.

## How good the instrument is (E1, E3-E7: ground truth first)

The verifier compares the weights of a model and a candidate parent tensor by tensor (median
cosine of sampled values over aligned tensors), read with HTTP range requests, so a 70B model costs
megabytes, not gigabytes. Its thresholds were fixed on development negatives only (largest
unrelated score + 5 SD: plain 0.0255, permutation-invariant 0.0662, aligned 0.0388) and then
judged once on fresh pairs.

| | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | Derivatives vs independent same-architecture models, fresh: TPR >= 95% at FPR <= 1% | **supported** | TPR 24/24, FPR 0/18 across six families (decoder, encoder, ViT, CNN, diffusion U-Net, speech; the last two blind) |
| H4 | A function-preserving permutation hides lineage, and weight matching restores it | **supported** | hidden 3/3, restored 3/3 |
| H5 | 4-bit quantised derivatives still verify; students distilled from scratch do not | **supported** | q4_0 6/6 verified; distilled students 0/12 related by weights |
| H6 | Sampled verdicts equal whole-file verdicts (>= 98%) | **supported** | 75/75 fresh E1 pairs and 50/50 Hub pairs |
| H10 | Weight similarity tracks shared initialisation, not shared data | **supported** | shared initialisation 4/4 related; same data, other initialisation 0/2 |
| H11 | A final model verifies against its own checkpoints from step 10k on, never another run's | **supported** | all own checkpoints related; other runs 0 |
| H7 | The candidate pool holds the true parent (>= 90%) | **not supported** | E1 derivatives 20/24; Hub models 76/199 |
| H8 | Direction (which model is the parent) is right on >= 90% of decided pairs, by Hub dates (D1) and by weight kurtosis (D2) | **not supported** | D1 183/193 (94.8%); D2 35/39 decided (89.7%), 20/59 undecided |
| H9 | Behaviour separates distilled students from same-task independent students (fresh) | **not supported** | TPR 3/12 at FPR 2/12 |
| H12 | Behaviour on off-task inputs separates them | **not supported** | TPR 9/20 at FPR 3/20 |
| H13a | Candidate pool v2 holds the confirmed parent of >= 90% of fresh Hub models | **supported** | 196/216 (90.7%, interval 86.1-93.9%); E2's pool 93/216 |
| H13b | Pool v2 holds more of them than E2's pool | **supported** | 104 found only by v2, 1 only by E2's pool; McNemar p = 2.6e-30 |
| H14 | Behaviour by shared mistakes, one rule for all families, fresh students: TPR >= 90% at FPR <= 5% | **not supported** | error agreement above 0.5965 (dev): TPR 0/20, FPR 0/20 |
| H14b | The same, with a rule per family (text decoder, vision CNN) | **not supported** | text decoder TPR 5/5, FPR 0/5; vision CNN TPR 5/5, FPR 5/5 |
| H15 | Direction by D3 (dates across owners or > 30 days; names within one release; kurtosis last): >= 95% right, <= 10% undecided, fresh | pending | fresh sample running; on E2 (dev) right on 184/189 decided, 10/199 undecided |
| H16 | Behaviour calibrated against reference students trained the same way: TPR >= 90% at FPR <= 5%, fresh | pending | dev rule fixed: drop > 6.3% (dev gap 5.7 points); fresh students wait for GPU quota |
| H16b | The same, with text-encoder and vision-transformer students trained long enough to learn the task | pending | waits for GPU quota |
| H17a | Structural GGUF alignment (names ignored) agrees with name alignment on >= 98% of fresh pairs | pending | fresh sample running |
| H17b | Structural alignment makes >= 50% of E2's 32 declared GGUF pairs with no name match comparable | pending | E2s running |
| H18 | Behaviour confirms >= 90% of the declared Hub pairs weights cannot compare (text models <= 14B) | pending | runs only if H16 or H16b is supported |

H2, H3, H7-H9, H12, H14 and H14b failed their bars; they are reported as they came out, and never
reworded. H2 and H3 are measurements of the Hub (its declarations are more honest than the hypotheses
guessed). Every other failed hypothesis has a follow-up, designed from the failure's cause in the data
and written into [PLAN.md](PLAN.md) before any of its data existed: H13 for H7 (supported), H15 for
H8, H16 and H16b for H9, H12, H14 and H14b, H17 for the unverifiable bases, H18 for behaviour above
2B parameters. H13 was judged on 360 Hub models no rule was chosen on, drawn before any was read;
H15 and H17a on a new sample of 360 (seed 20261118), drawn the same way.

## Every caveat, measured

Each weakness found along the way was either fixed and re-tested on fresh data, or measured.

- **The candidate pool (H7).** E2's pool missed the declared parent of 123 of 199 Hub models. Each
  miss has a recorded cause ([`results/h7_misses.json`](results/h7_misses.json)): 59 parents lie
  outside the 100k-model frame, 17 carry another family label, 16 differ in parameter count by
  more than 25% (packed quantised formats), 16 have no dimensions and differ by more than 2%,
  8 have no parameter count, 7 were ranked below the top 30. Pool v2
  ([`pedigree/pool_v2.py`](pedigree/pool_v2.py)) answers each cause without reading the model's
  own declaration: it adds the 23,649 bases other frame models declare, matches renamed
  repositories, ranks by name as well as by popularity, counts parameters without buffers, and
  searches the Hub once for the model's own name. On a fresh sample it holds the confirmed parent
  of 196 of 216 models (H13a, supported), against 93 for E2's pool, at a cost of 54 candidates
  per model instead of 26.
- **File readers.** 159 comparisons in the audit ended in a reader error. Three were bugs, now
  fixed and tested (MLX 3- and 6-bit values that straddle 32-bit words; MLX per-layer bit widths;
  GGUF int8 tensors). Scored again (E2r), 60 of them now compare and no verdict changes (H2
  106/367, H3 3/228, H7 b 77/200). The other 99 used GGUF tensor types the published `gguf`
  package does not list; see the fork types below.
- **GGUF names and old PyTorch files (amendment 2026-11-17).** Recent multimodal checkpoints nest
  the language model (`model.language_model.layers.N`), so llama.cpp's name table matched almost
  nothing in GGUF repackagings of them; names are now also matched with leading parts dropped,
  only where shapes agree and the match is unique. PyTorch files from before torch 1.6 (a series of
  pickles, not a zip) are read with the same restricted unpickler, and transposed 2-D weights
  (TensorFlow-converted BERT checkpoints) are read in row order. Scored again (E2g, 1,677 GGUF
  comparisons; E2p, 5 legacy bases): with all re-reads, declared bases the weights confirm rise
  from 214 to 243 models and unverifiable ones fall from 84 to 55; no verdict changes (H2
  106/375, H3 3/257, H7 b 77/230).
- **GGUF tensor types newer than the `gguf` package (E2f, amendment 2026-11-17 (4)).** Five are
  decoded from their public sources only, each tested byte for byte against that source's own C in
  CI: Q2_0 (42) and F8_E4M3 (43) from llama.cpp master, PrismML's PQ2_0 (142) and PTQ1_0 (143), and
  ik_llama.cpp's row-interleaved IQ4_K_R4 (339). Scored again, all 4 declared pairs that used them
  compare 24 tensors and are related (one, a ternary model, only by the permutation-invariant score:
  plain 0.004, invariant 0.743), and 91 of 103 pool comparisons now compare: declared bases the
  weights confirm rise to 246 and unverifiable ones fall to 52; no verdict changes (H2 106/378,
  H3 3/261, H7 b 78/234). Types 45, 101 and 103 appear in no public source we could find; the 11
  pool comparisons that use them say so. The one pool comparison with type 339 ran before its
  decoder existed and is being scored again.
- **The 52 still unverifiable (H17).** Their 54 declared pairs: 32 GGUF files whose architecture
  llama.cpp's name table does not know, so no tensor name maps; 10 that failed to re-read (network);
  7 other formats (a random tiny model with no shared shapes, a draft head, a repack, an NPU build,
  no weight files); 5 gated or private. H17 aligns GGUF tensors by block index and unique shape when
  names fail, validated against name alignment on fresh pairs (H17a), and applies it to the 32
  (H17b, with the 10 re-read).
- **Thresholds.** Under the pre-registered sensitivity rule (the midpoint between the classes
  instead of max + 5 SD) three verdicts change: H3 becomes supported (15/227 contradicted) and
  H10, H11 become not supported. The published rule is max + 5 SD; both are in
  [docs/RESULTS.md](docs/RESULTS.md).
- **Hub drift.** E2 took four passes (2026-11-08 to 2026-11-13). Pass 2 could not mount pass 1 and re-read 457
  models independently: 429 were identical, and the 28 that changed did so in their commit (13),
  declaration (12), pool (8) or another field (3); each model keeps its first reading
  ([`results/e2_rereads.json`](results/e2_rereads.json)).
- **Lost models.** A model lost to the network was retried in a later pass; the audit was
  recorded only when all 600 were read. Six losses in pass 3 were two bugs (unescaped spaces in
  file names; an index naming a shard the repository lacks), fixed before pass 4.
- **Behaviour.** Not supported on ground truth (H9, H12, H14, H14b) and applicable to 0 of 94 Hub
  pairs; it is reported as its own category and never moves H3. H14 tested what only a distilled
  student should inherit: the teacher's mistakes, and how it spreads probability over wrong
  answers (E5e; 20 fresh distilled and 20 fresh independent students, trained after the rule was
  written). One rule for all families finds nothing (TPR 0/20): the families sit on different
  scales. With a rule per family (H14b), dark-knowledge divergence identifies every fresh
  text-decoder student correctly (5/5 distilled, 0/5 independent) but admits every vision-CNN
  student (5/5 and 5/5; its dev margin was 0.017), and no rule exists for the text encoder or the
  vision transformer, whose students barely learn the teacher's task. The cause is in the data:
  within each training batch the vision-CNN students do separate (fresh distilled 0.091-0.105
  against independent 0.128-0.143), but the whole fresh batch sits below the threshold chosen on
  the dev batch (0.174); an absolute threshold does not carry from one batch to the next. H16
  scores each student against reference students trained the same way (its rule, a drop of more
  than 6.3% below the lowest reference, was fixed from dev before any fresh student exists); H16b
  trains text-encoder and vision-transformer students for 20,000 steps, to a competence floor set
  in advance. Both run on GPU when the quota is free.
- **Behaviour above 2B parameters (H18).** 66 of the 94 declared pairs weights could not compare
  have a model above 2B parameters. Behaviour runs on them (text models up to 14B sharing a
  tokenizer) only if H16 or H16b is supported: a behaviour result on a large model proves nothing
  until the behaviour check itself has passed on ground truth. Most of these pairs are GGUF files
  that H17 may make comparable by weights instead.
- **Direction (H15).** Hub dates (D1) are right on 183 of 193 pairs; 9 of the 10 misses are one
  organisation's joint releases, where the derived repository was created up to 7.9 days before
  its base (Qwen, Gemma, LiquidAI, nomic), and one is a base re-created years later. Weight
  kurtosis (D2) is right on 35 of 39 decided pairs and leaves 20 of 59 undecided: it is right on
  every pre-training checkpoint pair, but short fine-tunes move it by less than 0.005 either way and
  LoRA merges not at all. D3 uses dates across owners or more than 30 days apart, names within one
  release, and kurtosis last; on E2 (dev) it is right on 184 of 189 decided pairs, and H15 judges
  it on a fresh sample.

## Use it

```bash
pip install -e .
pedigree verify hf:org/model                      # against its declared base_model
pedigree verify hf:org/model --base hf:org/base   # against a base you name
```

Verdicts are *related*, *unrelated* (with enough tensors compared) or *not comparable*. Nothing
downloaded is executed: PyTorch checkpoints are read by a restricted unpickler. A browser version
is on the [project site](https://aaditripi18.github.io/Pedigree/check.html): it runs in the page,
reads only from huggingface.co, and takes shareable links, e.g.
[DialoGPT-small vs GPT-2](https://aaditripi18.github.io/Pedigree/check.html?model=microsoft/DialoGPT-small&base=openai-community/gpt2).

## Reproduce

```bash
pip install -e ".[dev,experiments]"
pytest
python scripts/analyse.py      # every verdict, from results/ledger.jsonl
python scripts/build_site.py   # the site, from results/verdicts.json
```

CI runs the tests and fails if `results/verdicts.json`, `docs/RESULTS.md` or the site differ
from what the ledger gives; every push to `main` that changes `site/` redeploys the site.

The heavy runs (fine-tuning ground truth on GPU, the 600-model audit on CPU) ran on Kaggle with
[`scripts/kaggle_run.py`](scripts/kaggle_run.py), which uploads only committed code and stamps
every result with its commit. Most ran on the project's own account; when its weekly GPU quota
ran out, a few jobs ran with the same committed code on friends' accounts and were shared back
(each is named in `scripts/kaggle_run.py`).

## Licence

MIT.
