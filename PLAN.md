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

## Limitations designed out before any run (2026-11-04)

Stowaway and leakcheck each finished with a list of limitations that later
phases had to remove. Pedigree builds the answer to every one we can foresee
into the first design. Each row is part of the plan, not an afterthought.

| Foreseeable limitation | Built-in answer | Measured by |
|---|---|---|
| Verifier tested on one model family (LLMs only, as in AWM/HuRef) | Ground truth stratified by family: text encoders, decoders (small LLMs), vision transformers and CNNs, speech, diffusion UNets | E1, per-family TPR/FPR |
| Ground truth that is itself uncertain | Two kinds only: (a) pairs we create ourselves (fine-tune, LoRA merge, quantise, prune, distil small models on CPU/Kaggle), whose lineage is exact by construction; (b) official derivatives an organisation publishes of its own base | E1 |
| Threshold chosen on the data it is judged on | Dev/fresh split from the start; a full ROC and AUC, not one operating point; the fresh set is used once | E1 |
| Sampled tensors may not represent the whole file (Stowaway E10) | Sampling checked against whole files on a subset before the audit: same verdict required | E1b, H6 |
| Survey biased to the most-downloaded models (Stowaway E10 B) | The audit samples three download tiers (top 1k, 1k-10k, 10k-100k) at random, frozen before reading | E2 |
| "Discovered" means a relative, not the parent (Stowaway E8) | Relations kept apart: parent, descendant, sibling, copy. Direction decided separately (Hub creation dates, and weight-statistics direction as in MoTHer) and its accuracy measured on ground truth; the audit never calls a relative a parent without it | E1c, H8 |
| The true parent may not be among the candidates | Candidate recall measured on ground truth (is the true parent in the pool?); pool widened by config, tokenizer and embedding-matrix fingerprints, not only the 20 most downloaded | E1d, H7 |
| Hub rate limits and slow networks lose models (Stowaway E8) | Retries with Retry-After from the first line of code; every response cached; Kaggle CPU for bulk; a run is reported only if no model was lost to the network | all |
| Quantised uploads (GGUF, 8/4-bit) cannot be compared to float bases | Dequantisation for GGUF Q8_0/Q4_K and 8-bit safetensors before comparison | E4, H5 |
| Lineage hidden by permutation, scaling or rotation | Attacks and the invariant defence are part of the first version, not a later fix | E3, H4 |
| Shapes differ (pruned, distilled, resized vocab) | Partial alignment on the tensors that do match, reported as such; "unverifiable" is a verdict, never silently "unrelated" | E4 |
| Proportions from one sample with no uncertainty | Every audit proportion with a 95% interval (Wilson / bootstrap); sampling seed recorded | E2 |
| The Hub's own relation tags (finetune / adapter / quantized / merge) unchecked | Each declared relation type is checked against what the weights show | E2, H3 |
| A finding nobody can use | CLI (`pedigree verify hf:model [--base hf:base]`) and a browser checker reading only headers and tensor samples | release |

Added hypotheses (same rules as H1-H5):

- **H6** On the whole-file subset, the sampled verdict equals the whole-file verdict
  for >= 98% of pairs.
- **H7** For ground-truth derivatives with a public parent, the widened candidate
  pool contains the true parent in >= 90% of cases.
- **H8** Direction (which of two related models is the parent) is decided
  correctly for >= 90% of ground-truth pairs where it is decided at all, and the
  share left undecided is reported.

## Second instrument: behaviour, where weights cannot be compared (2026-11-04, before any run)

A weight comparison needs shared tensors. A distilled student, a resized model or a
different architecture shares none with its source, so a weight-only verifier can
only answer "unverifiable" there; left alone that would become a limitation. Pedigree
therefore has a second, independent instrument for exactly those cases:

**Behavioural similarity.** Both models run on a fixed probe set (texts for text
models, images for vision models, audio for speech; frozen and committed, never
used for training any derivative), and their output distributions are compared
(mean Jensen-Shannon divergence of next-token / class / token distributions over a
shared output space, and agreement of top-1 predictions). A distilled student
imitates its teacher's outputs, so behaviour can show lineage where weights cannot.
Behaviour is weaker evidence than weights (two models trained on the same task can
agree for honest reasons), so it is calibrated against same-task independent models,
not just random ones, and reported as a separate verdict.

- **E5. Behavioural lineage.** Positives: students distilled from each E1 base (and
  public distilled models whose teacher is documented, such as DistilBERT from
  BERT-base and DistilGPT-2 from GPT-2). Negatives: independently trained models of
  the same size and task. Dev/fresh split as in E1.
- **H9** Where weights cannot be compared, behavioural similarity separates distilled
  students from independently trained models of the same task with TPR >= 90% at
  FPR <= 5% on the fresh set.

The audit's verdicts become: verified by weights; verified by behaviour only;
contradicted; unverifiable (neither instrument applies). The last is reported with
its share, so no model silently drops out.

## Amendment 2026-11-05: ground truth corrected before any comparison

Made after the smoke test of the fine-tuning script and before any pair of trained
weights was compared (no similarity has been computed on any E1 pair).

**Two "independent" negatives were not independent.** In the frozen list of 2026-11-04:

- `distilgpt2` was listed as independent of `gpt2`. DistilGPT-2's student was
  initialised from GPT-2's own layers (the transformers distillation recipe that
  produced it copies every other teacher layer), so it is a *derivative*. It moves to
  the public distillation positives (E4/E5, partial shapes).
- `pythia-160m-deduped` was listed as independent of `pythia-160m`. Pythia runs share
  their initialisation seed, so the two start from the same weights and differ only in
  training data. It moves to the shared-origin study below.

**Replacement negatives (same architecture, trained from scratch, different seed).**

- gpt2 (dev): Stanford CRFM's Mistral GPT-2 small runs `alias-x21`, `battlestar-x49`,
  `caprica-x81`, `darkmatter-x343`, `expanse-x777`: five independent from-scratch
  trainings of the GPT-2 small architecture.
- pythia-160m (fresh): PolyPythias `pythia-160m-seed1` ... `seed9` (arXiv 2503.09543),
  nine trainings with different seeds.

**PyTorch checkpoints are read, not only safetensors.** These models (and many Hub
models) exist only as `pytorch_model.bin`. `pedigree/torchzip.py` reads a PyTorch zip
checkpoint with range requests through a restricted unpickler that executes nothing
(tested against a malicious pickle). Without it the Hub audit would silently skip every
pickle-only model.

**E6 (new). What does weight similarity detect: shared initialisation or shared training?**
PolyPythias also publishes runs that vary one factor: `data-seed1..3` (data order
varied, initialisation fixed) and `weight-seed1..3-corrected` (initialisation varied,
data order fixed; the uncorrected weight-seed models had a training error and are not
used). Which runs share `pythia-160m`'s initialisation is decided by byte equality of the
published `step0` weights (`experiments/init_identity.py`, `results/e1_init_identity.json`),
never by the trained weights. These pairs are neither positives nor negatives for H1;
they are reported separately.

- **H10** Weight similarity tracks shared initialisation, not shared data: runs that share
  `pythia-160m`'s step0 weights score above E1's threshold against it on >= 2/3 of pairs,
  and runs with a different initialisation but the same data order on <= 1/3.

**E7 (new). How far back along a training run is lineage visible?** Each Mistral run
publishes about 600 checkpoints from step 0 to 400,000. Positives: a run's final weights
against its own earlier checkpoints (steps 0, 1k, 10k, 50k, 100k, 200k, 300k), exact
ancestors by construction. Matched negatives: the same steps of the other four runs. The
result is a curve of similarity against training distance, and direction (H8) is judged
on these pairs too, since the earlier checkpoint is the parent.

- **H11** The final model is verified against its own checkpoints from step 10,000 on
  for all five runs, and no checkpoint of another run is verified at any step.
