# Pedigree: do models come from where they say they do?

## The problem

A model card's `base_model` field is how the Hugging Face Hub, licence checks,
procurement reviews and model-documentation rules learn what a model is built
on. Nobody checks it. A fine-tune of a model with a restrictive licence can be
re-uploaded under a permissive one with no parent named; a card can name a base
the model does not descend from. In Stowaway's E8 (23 popular float32 models),
only 5 declared a usable base, and one declared base shared 90% of its tensor
names and shapes but had a weight cosine of 0.07 with the model: a different
model under the same architecture.

## What already exists (read 2026-11-04)

Weight-based lineage detection is an active research area with strong methods:

- **AWM** (ICLR 2026, arXiv 2510.06738): training-free weight-matrix fingerprints
  for LLMs; the authors report perfect classification on 60 base/offspring and 90
  independent pairs, robust to SFT, continued pretraining and RL.
- **HuRef** (NeurIPS 2024, arXiv 2312.04828): the parameter-vector direction of an
  LLM survives fine-tuning; three permutation-invariant terms resist weight
  rearrangement attacks.
- **MoTHer** (ICLR 2025, arXiv 2405.18432): recovers model trees from weights
  (directed minimal spanning trees over weight distances); notes that most public
  models lack documentation.

Pedigree does **not** propose a better detector. It uses established methods,
credited, as instruments, and measures what none of these papers measure: **how
often the Hub's declared lineage is missing, unverifiable, or contradicted by the
weights**, across model types (not only LLMs), on a frozen random sample. The
verifier's accuracy is measured on ground truth before it is used, and its
known weaknesses (permutation, quantisation) are measured too.

## Questions

- Q1. Of the models that are verifiably derived from another public model, how
  many declare their parent?
- Q2. Of the models that declare a base, how many declare one their weights do
  not support?
- Q3. How accurate is a weight-based verifier on ground truth, across model
  families and the transformations real uploads apply (fine-tuning, LoRA merges,
  quantisation, pruning, distillation)?
- Q4. Can lineage be hidden by a function-preserving weight transformation, and
  does a permutation-invariant check recover it?

## Method

**Reading weights without downloading them.** As in Stowaway, safetensors headers
and the first bytes of selected tensors are read with HTTP range requests, so a
model costs kilobytes to megabytes, not gigabytes. Whole files only where a
check needs them.

**Verifier (instrument).** For a model M and a candidate parent P with the same
tensor names and shapes (aligned as in Stowaway's S5): the median, over sampled
tensors, of the cosine similarity of the weight values. Calibrated on ground
truth (E1) before use. A permutation-invariant variant (sorting-based or the
HuRef-style invariant terms, chosen in E1 dev) handles rearranged weights (E3).

**Candidate parents.** The declared `base_model` if any; otherwise the most
downloaded Hub models with the same `model_type` and dimensions (as in Stowaway
E8).

## Experiments

- **E1. Ground truth.** Pairs with known lineage: official derivatives (an
  organisation's fine-tunes and quantised releases of its own base models), LoRA
  merges, and independent models of the same architecture (negatives). Split dev
  (choose the rule and threshold) / fresh (judge once). Frozen lists committed
  before any weight is read.
- **E2. The Hub audit.** A frozen random sample of models from the most
  downloaded public safetensors models; for each: declared base (yes/no;
  downloadable; verified/contradicted by E1's rule) and discovered relatives.
- **E3. Hiding lineage.** Apply function-preserving neuron permutations (and
  scaling where the architecture allows) to known derivatives; measure the plain
  and the invariant verifier.
- **E4. Hard cases.** Quantisation (8-bit, 4-bit), pruning, distillation: where
  does verification stop working.

## Hypotheses (pre-registered; decision rules fixed before the runs)

- **H1** The verifier separates derivatives from independent same-architecture
  models on the fresh ground-truth set with TPR >= 95% at FPR <= 1%.
- **H2** In the Hub sample, at least 30% of models verifiably derived from another
  public model do not declare a base.
- **H3** In the Hub sample, at least 5% of declared bases are contradicted by the
  weights (median cosine below E1's threshold while a same-shape comparison is
  possible).
- **H4** A function-preserving neuron permutation drops the plain verifier below
  its threshold on >= 90% of derivatives, and the permutation-invariant check
  restores detection on >= 90%.
- **H5** 4-bit quantised derivatives are still verified (>= 90%); distilled
  students are not (<= 10%).

Each verdict is reported as supported or not supported with its numbers, never
reworded after the fact.

## Reproducibility discipline (as in Stowaway, leakcheck, Proof of Forgetting)

- Every number is appended to `results/ledger.jsonl` with the git commit; the
  results document and the website are generated from it.
- Model lists are frozen and committed before any weight is read; fresh sets are
  used once.
- Any change to a rule after seeing data is written here, dated, with the data it
  rests on, and judged on a fresh set.
- CPU only; download-heavy runs go to a Kaggle CPU session.

## Ethics

Pedigree reports what weights show about lineage, not intent. A missing
`base_model` is usually an omission, not a deception; results are reported
in aggregate, and individual models are named only as worked examples of a
pattern, with the evidence.
