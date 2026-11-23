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

## Amendment 2026-11-05 (2): E3 tests the strongest permutation, before any E3 result

A smoke run (3 training steps, scratchpad only, not a result) showed a structural fact
that follows from the code alone: permuting only feed-forward hidden units leaves every
attention tensor untouched, so the plain median falls only part way. A real attacker
would permute the residual stream as well. E3 therefore has two attacks and two defences:

- **Attack (a), FFN only:** the hidden units of every feed-forward block (as before).
- **Attack (b), full:** (a) plus one random permutation of the residual stream applied
  consistently to embeddings, position embeddings, every LayerNorm, the input axis of
  every layer that reads the stream and the output axis of every layer that writes it
  (attention head space untouched). LayerNorm is permutation-equivariant, so the model
  computes the same function; each attacked model is checked by running the real model
  forward before and after on random inputs (max abs difference reported), and a model
  that fails the check is reported, never used.
- **Defence (i), one-sided invariant:** median off-diagonal Gram cosine (as before).
- **Defence (ii), align then compare:** recover the residual permutation by matching the
  columns of the embedding (or patch-embedding) matrix to the candidate parent's with the
  Hungarian algorithm on their correlations (weight matching, as in Git Re-Basin), undo
  it, then apply (i). Independent models are matched the same way, so whatever matching
  adds to unrelated pairs is measured on the dev negatives, not assumed away.

H4 is judged on attack (b) with defence (ii); attack (a) and defence (i) are reported
beside it. Families where attack (b) is implemented: text encoder, text decoder, vision
transformer (the regular transformer stacks); CNN, speech and diffusion get attack (a)
and the channel permutation already in `transforms.py`, reported as such.

## Amendment 2026-11-07: decision rules, H6-H8 and the audit protocol, before any verdict

Written before any hypothesis verdict was computed and before E2 read any weight.
**What had been seen.** The run logs had printed the plain, invariant and aligned scores of
every E1 pair (dev and fresh) for text encoders, text decoders, vision transformers and
CNNs, and the E6 and E7 scores. Speech and diffusion pairs had not been compared yet, so
they are the only E1 pairs on which the rule below is tested blind; this is stated beside
every verdict.

**Threshold rule (every score: plain, invariant, aligned; one threshold per score, pooled
over families so it applies to Hub models of any family).** t = (highest score of any dev
negative) + 5 x (standard deviation of the dev negatives' scores). A pair is *related*
(lineage evidence) when its score is above t. Why this rule: the question every hypothesis
asks is whether lineage is visible above chance, and the audit makes thousands of
comparisons (each model against its candidate pool), so the margin is wide (5 SD) rather
than tuned. Sensitivity rule, reported beside every verdict and never substituted for it:
the midpoint between the highest dev negative and the lowest dev trained positive.

**Verdict of the weight verifier.** *Related* if plain > t_plain, or the invariant score
(defence i, or defence ii where implemented) is above its own threshold; *unrelated* if
every computed score is at or below its threshold while at least 8 tensors were compared;
otherwise *not comparable* (fewer than 8 shared tensors), which is never counted as
unrelated.

**H1** positives: the trained kinds (ft_short, ft_long, lora_r8, lora_r64) of every family
on fresh; negatives: fresh scratch and independent models. ROC AUC and TPR/FPR with Wilson
95% intervals. Fine-tunes move weights little, so these positives are easy; the hard end is
measured by E7 (100k-399k further pretraining steps) and E4, reported as a curve.

**H4** on fresh, attack (b) families: plain at or below t_plain on >= 90% of attacked
models, defence (ii) above t_aligned on >= 90%. **H5** on fresh: q4_0 related (plain) for
>= 90% of families; the students distilled in E5 (random initialisation) related by weights
for <= 10% of students. DistilGPT-2 and DistilBERT, initialised from their teachers' layers,
are reported separately.

**H6, whole files.** Whole-file score: the same median cosine, but over *every* aligned
tensor with >= 4096 elements, each read in full. Subset: every fresh E1 pair, and 50 E2
pairs drawn at random (seed 20261107) from the pairs E2 compared whose two models are each
<= 2 GB. H6 holds if the related/unrelated verdict under the plain rule agrees on >= 98%.

**H7, candidate recall.** The candidate pool of a model (no use of its own declaration):
frame models (results/e2_frame_meta.jsonl.gz) with the same `model_type` and a parameter
count within 2% (from the Hub's safetensors metadata; GGUF parameter count for GGUF repos);
ranked first by how many frame models declare them as a base, then by downloads; the top 30
are compared, plus the declared base when one exists. Ground truth with a public parent:
(a) the fresh E1 derivatives and their transforms, parent = the fresh base; (b) E2 models
whose declared base is related by weights, parent = that base, with the declaration hidden.
H7 holds if the true parent is in the pool for >= 90% of (a) and of (b) separately.

**H8, direction.** Two rules, both judged: (D1, used by the audit) the relative created
earlier on the Hub is the parent; judged on E2 pairs whose declared base is related by
weights (declared base = parent). (D2, weights only, as in MoTHer) the model with the
lower median excess kurtosis over the compared tensors is the parent; undecided when the
difference is below the margin m chosen on E1 dev (the smallest m with no dev error);
judged on fresh E1 trained pairs and all E7 pairs (earlier checkpoint = parent). H8 holds if
each rule is right on >= 90% of the pairs it decides; the undecided share is reported.

**E2 protocol.** Every sampled model gets exactly one category, with the reason:
- *weights not readable*: no safetensors, PyTorch zip, GGUF or MLX weights (ONNX-only,
  pickle-only non-zip, code-only, gated without access, deleted);
- *adapter*: the repo holds only an adapter (a weight delta); checked for shape consistency
  with its declared base, not counted in H2 or H3;
- otherwise, the declared base (if any) is compared: *declared, related*, *declared,
  unrelated* (H3's "contradicted"), or *declared, not comparable* (then the behaviour
  instrument where both are text models with the same tokenizer or image classifiers with
  the same label space and <= 2B parameters each; else unverifiable, reason recorded);
- every readable model is compared with its candidate pool; a related candidate created
  earlier is a *discovered parent*, later a *discovered descendant*; identical sampled bytes
  are a *copy*.
- **H2** denominator: models with a related earlier relative (declared or discovered);
  numerator: those that declare no base. **H3** denominator: declared bases that were
  comparable; numerator: unrelated. Both judged on the pooled 600-model sample as written;
  the estimate weighted by tier size (1k / 9k / 90k) and per-tier values are reported with
  Wilson 95% intervals. The Hub's relation tag (finetune, adapter, quantized, merge) is
  checked against the weights: "quantized" is consistent when plain >= (lowest dev q4_0
  plain - 5 SD of the dev q4_0 values).
- A model lost to the network is retried in a later pass; the audit is reported only when
  none is lost.

## Amendment 2026-11-07 (2): audit details, before E2 or H7 read any weight

Fixed while writing the audit code, from reasoning about the rules, not from data.

- **Candidate pool, corrected.** The 2% parameter rule above would miss the parent of
  any fine-tune that replaced its classification head (a ResNet-18 trained for 10 classes
  has 4% fewer parameters than its 1000-class base). The pool is instead: same family key
  (`model_type`; else the GGUF architecture; else the library), parameter count within
  25%, and identical dimensions from `config.json` (hidden size, number of layers, feed-
  forward size; timm's `architecture`); a candidate or model without readable dimensions
  falls back to parameter count within 2%. Ranking and the top 30 as before. A model's
  parameter count is the Hub's safetensors or GGUF count, or, where that is missing or
  counts packed values (MLX), the number of values its own header describes.
- **Gated models** are read through a public repository whose weight files have exactly the
  same SHA-256 as the gated repository's (the Hub publishes checksums of gated files); the
  report names the copy. Without such a copy a gated model is *not readable (gated)*.
- **Which files.** Adapter files (`adapter_model.*`) are never taken as model weights; a
  repository with only an adapter is an *adapter*. A diffusers repository is read through
  its denoiser (`unet/`, else `transformer/`). A PyTorch file that is not a zip checkpoint
  (the pre-2020 format) is *not readable (legacy pickle)*.
- **The invariant check** (defence i) is run on a declared pair only when the plain score
  is below 0.2 (a superset of every plain threshold the rule can produce from E1 dev), on
  matrices of at most 32M values each; skipped matrices are recorded.
- **A copy** is a related candidate whose every sampled value is identical; copies are
  reported apart from parents and descendants, and a copy created earlier is not counted
  as a parent for H2 (it is the same model, uploaded twice).
- **Network.** A model or candidate that could not be read because of the network (not
  an HTTP 401/403/404 or a format error) is retried in the next pass.

## Amendment 2026-11-07 (3): frozen configs, direction data, H9's rule, before E5 or E2 ran

- **Frozen dimensions.** The Hub's listing leaves `model_type` out for about a third of
  transformers models (35,032 of 59,473 in the frame), which would hide them from every
  pool. Every frame model's `config.json` is therefore fetched once and reduced to its
  `model_type` and dimensions (`results/e2_frame_configs.jsonl.gz`); the pool uses it.
- **Direction data.** E7 ran before the kurtosis statistic existed, so its 35 same-run
  pairs are compared again with the plain verifier, recording the kurtosis gap (`E7k`).
  The plain scores must equal E7's (same tensors, same bytes); any difference is reported.
- **H5's students.** E5 also runs the weight verifier on every student/teacher pair.
- **H9's rule (dev only).** Of three statistics (top-1 agreement, 1 - JS divergence, linear
  CKA), the one with the widest dev gap between distilled students and same-task independent
  students (lowest positive minus highest negative) is used, with the threshold at the
  middle of that gap. If no statistic separates dev, agreement is used with the threshold
  giving the highest dev TPR at dev FPR <= 5%. H9 holds if, on fresh, TPR >= 90% on the
  distilled students and FPR <= 5% on the same-task independent students.

## Amendment 2026-11-08: two file-selection gaps found by a smoke test on non-sample models

A smoke test of the audit on five frame models *outside* the sample (scratchpad only, not a
result) found: (1) repositories that keep weights only in subfolders without a diffusers
layout (bundles of several networks) were "not readable"; they are now read through their
largest weight unit (a folder's sharded set or a single file), the main network; (2) LoRA
weights saved under a name other than `adapter_model.*` were read as a model; a repository
whose tensor names are mostly LoRA tensors is now an *adapter*. MLX tensors with an experts
axis (mixture-of-experts) are dequantised too.

## Amendment 2026-11-08 (2): GGUF names in the candidate pool, before E2 or H7 ran

Freezing every frame model's `config.json` (amendment 2026-11-07 (3)) showed that its
stated reason was mostly wrong: of the 35,032 transformers-tagged frame models without a
`model_type` in the listing, 34,490 are GGUF repositories, which have no `config.json` at
all (473 have one; 67 gained a `model_type` from it). They are keyed by their GGUF
architecture name, and llama.cpp spells those differently from Hugging Face (`qwen35moe`
vs `qwen3_5_moe`, `gemma3` vs `gemma3_text`), and files Mistral and Mixtral under `llama`,
Qwen2.5-VL under `qwen2vl`, DeepSeek V2/V3 under `deepseek2`. Left alone, a GGUF upload's
pool would never contain its Hugging Face parent. Family keys are therefore normalised
(lower case, separators and a trailing "text" removed) and those four aliases applied. The
frozen file is kept: it still supplies every frame model's dimensions (27,717 have them).

## Amendment 2026-11-08 (3): H12, behaviour on off-task inputs, after H9 was decided

H9 is not supported (fresh TPR 3/12 at FPR 2/12 with agreement; no statistic separated dev),
and stays so. Its numbers show why: each family has its own level of teacher agreement (1 - JS
about 0.15 for the masked LM, 0.85 for the CNN), and in-distribution probes reward any model
that learned the task, so same-task independent students sit close to distilled ones. Both
points suggest a new hypothesis, written before any of its data exists:

- **H12** On off-task inputs, a distilled student agrees with its teacher more than the
  family's same-task independent students do: TPR >= 90% on new distilled students at
  FPR <= 5% on new independent students (fresh).
- **Off-task probes (frozen).** Images: 1,000 images of N(0, 1) noise in the model's
  normalised input space, `torch.Generator` seed 0. Text: E5's 256 WikiText-2 test lines with
  their words shuffled (Python `random.Random(0)` per line), so the tokens are familiar and
  the text is not. Masked positions as in E5.
- **Score.** The statistic for the student minus the highest value of the same statistic among
  the family's reference independent students on the same split (indep seeds 0-2; a dev
  independent student is scored against the other two).
- **Rule (fixed on dev, seeds 0-2 only).** Of agreement, 1 - JS and CKA, the one with the
  widest dev gap between distilled and independent scores, threshold at the middle of the
  gap. If none separates dev: the statistic and threshold with the highest dev TPR at dev
  FPR = 0 (12 negatives), the widest gap breaking ties.
- **New data.** Fresh-split students with seeds 3-7 (five distilled, five independent per
  family; 20 and 20), trained exactly as E5's, after this amendment. They are judged against
  references seeds 0-2. The same students are also scored on E5's original probes (recorded as
  `E5n`, so H9's runs are untouched), to show how much the change of probes, rather than the
  new students, explains.
- Ledger: `E5o` (off-task probes, seeds 0-7, both splits), `E5n` (original probes, seeds 3-7).

## Amendment 2026-11-13: reader fixes, candidate pool v2 and H13, after every earlier verdict

All twelve earlier verdicts stand as recorded; nothing below changes them.

**Reader fixes (E2r).** 159 of E2's comparisons ended in a reader error, not a verdict. Three
were reader bugs: MLX 3- and 6-bit values straddle 32-bit words (they were unpacked as if not);
MLX per-layer bit widths are written under the layer's path, sometimes with `model` and
`language_model` in the other order; GGUF int8 tensors are stored as they are, not quantised.
The fourth cause is GGUF tensor types outside llama.cpp's published list (fork formats, types 42,
45, 101, 142, 339), which no reader can decode from a public specification; they now say so.
`E2r` scores the 159 again with the fixed readers. E2 stays as recorded; `scripts/analyse.py`
reports beside it E2 with these scores, and says whether any verdict would change.

**Candidate pool v2.** E2's pool held the weight-verified declared parent for 76 of 199 Hub
models (H7 b). Every miss has a recorded cause (`results/h7_misses.json`). Pool v2
(`pedigree/pool_v2.py`) answers each cause without reading the model's own declaration: the
frame plus every base that another frame model declares (`results/e2_bases_meta.jsonl.gz`,
23,649 bases fetched 2026-11-13; renamed repositories matched by their current name); the union
of the children-ranked and name-ranked top 30; parameter counts without integer buffers; size
and family rules loosened only where packed formats, missing dimensions or the model's name
justify it; and one Hub search for the model's own name. It was designed on E2's 199 models and
H7 (a)'s 24 derivatives, so its numbers there are dev numbers: 174/199 and 24/24, at a mean of
54 candidates against E2's 26.

**H13, judged on fresh data only.** A new sample (`results/h13_sample.json`, committed
2026-11-13 before any of it was read): 120 frame models per download tier that declare a base
and were not in E2's sample (seed 20261113). Each is compared by weights with its declared
bases exactly as in E2, and its Hub search is recorded (`experiments/h13_fresh.py`, ledger
`H13`). The ground truth is each declared base the weights confirm (not a byte-identical copy),
as in H7 (b). Both pools are rebuilt from frozen metadata after the reading.

- **H13a** Pool v2 holds the weight-verified declared parent for >= 90% of these models.
- **H13b** Pool v2 holds more of them than E2's pool: exact one-sided McNemar test on the
  discordant pairs, p < 0.05.

Wilson intervals, per-tier values and the mean pool size are reported. The audit's E2 numbers
are not recomputed with pool v2: E2 is reported with the pool it ran with.

## Amendment 2026-11-17: GGUF names in multimodal checkpoints, after every earlier verdict

All earlier verdicts stand as recorded; nothing below changes them.

Of the 84 E2 models whose declared base could not be compared by weights, most are GGUF files
(llama.cpp's format) declared against a safetensors base. `ggufremote.hf_names` maps GGUF names
back with llama.cpp's own table, which knows a language model's plain names
(`model.layers.3...`). Recent multimodal checkpoints nest the language model
(`model.language_model.layers.3...`), so almost nothing mapped: on six such pairs, 0 or 1
tensors aligned. The mapper now also tries a name with its leading parts dropped, keeping such a
match only when the shapes are equal and no other tensor claims the same GGUF tensor that way (a
vision tower's `layers.3` cannot stand in); names that mapped before map exactly as before
(`tests/test_ggufnames.py`). On the six pairs, 399 to 749 tensors now align. No threshold or
rule changes.

**E2g** (`experiments/e2_gguf_rescore.py`) scores again every comparison of E2's last run, after
E2r, that involves a GGUF file and compared fewer than `MIN_COMPARED` tensors without an error,
declared bases and candidate-pool pairs alike. E2 stays as recorded; `scripts/analyse.py`
reports beside it E2 with the E2r and E2g scores, and says whether any verdict (H2, H3, H7)
would change. A changed verdict is reported as a change, not as the verdict. The first E2g run stopped after 467 of 1,677
comparisons on an empty Hub reply that one comparison should have recorded; the script now records
any such failure and resumes, skipping pairs already scored, and each pair's latest score counts.

**E2p.** PyTorch files saved before torch 1.6 are a series of pickles followed by raw storages,
not a zip; the reader refused them ("legacy pickle files are not read"), which left 6 declared
bases unreadable. `pedigree/torchzip.py` now reads them with the same restricted unpickler
(PyTorch's tensor records only, any other global refused; `tests/test_torchzip.py` includes a
legacy file carrying code). E2p (`experiments/e2_rescore.py --match "legacy pickle" --name E2p`)
scores again every E2 comparison that ended in that error, and is applied beside E2 as E2r is. Its
first run read 2 of the 5 pairs fully; in the other 3 (BERT checkpoints converted from
TensorFlow) every dense weight is stored transposed, which the reader left out, so only 2 large
tensors could be compared. A plain transpose of a 2-D tensor is now read whole and put in row
order (checked on `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`: 208 of 208
tensors equal `torch.load`'s); E2p is run again, and its latest run is the one applied.

## Amendment 2026-11-17 (2): H14, behaviour by shared mistakes, before any of its data

H9 and H12 stand as recorded (not supported). Agreement, Jensen-Shannon divergence and CKA
measure how alike two models' answers are, and a student trained independently on the teacher's
task learns many of the same answers. What only a distilled student inherits is the teacher's
idiosyncrasy: its mistakes, and how it spreads probability over the wrong answers.

**Statistics (E5e).** Over E5's probes, with the true answer at each position (the CIFAR-10
label; the masked token for a masked LM; the next token for a causal LM): *error agreement*, the
share of positions the teacher gets wrong where the other model gives the same wrong answer; and
*dark-knowledge divergence*, the Jensen-Shannon divergence of the two distributions once the
teacher's top class is removed and the rest renormalised. E5's own statistics are unchanged
(`tests/test_behaviour.py`).

**Rule, chosen on dev only.** E5e scores E5's dev students (seeds 0-2, four families, with the
public derivatives and independents E5 used). For each statistic, the dev gap is the smallest
distilled score minus the largest independent score (for the divergence, negated). The statistic
with the larger gap is used, with its threshold at the gap's midpoint (if no gap is positive: the
threshold that admits the most distilled students while admitting no independent one). Both
are written into `results/verdicts.json` from the dev run alone.

**Fresh data.** New students, never trained before this amendment: seeds 8-12 of each family's
fresh split, distilled and independent alike (`distill3-*`, GPU), 20 distilled and 20
independent; E5e-fresh scores only them.

- **H14** On the fresh students, the dev rule identifies distilled students with TPR >= 90% at
  FPR <= 5% (at most one independent student admitted), per-family values reported.

## Amendment 2026-11-17 (3): H14b, per-family rules, after E5e dev and before any fresh student is read

E5e dev (four families, seeds 0-2, with E5's public models): pooled over families, neither
statistic separates distilled from independent models (error agreement: smallest distilled minus
largest independent -0.564; dark-knowledge divergence -0.674). The families sit on different
scales, and the well-trained public GPT-2 replicas agree with GPT-2 more than any student does. H14
keeps its rule as written (error agreement, the fallback threshold, which admits no dev student)
and is judged on the fresh students unchanged.

Within a family the students separate where they learned the task. **H14b** uses a rule per
family, chosen on that family's dev students only (`results/h14b_rules.json`, written before any
fresh student was read): the statistic with the larger gap, threshold at its midpoint.

| Family | Statistic | Dev gap | Threshold |
|---|---|--:|--:|
| text decoder | dark-knowledge divergence (lower = distilled) | 0.127 | 0.500 |
| vision CNN | dark-knowledge divergence | 0.017 | 0.174 |
| text encoder | none separates (students agree with the teacher on 7% of positions) | -0.003 | - |
| vision transformer | none separates | -0.023 | - |

- **H14b** On the families with a dev rule (text decoder, vision CNN; 10 fresh distilled and 10
  fresh independent students), the rule identifies the distilled ones with TPR >= 90% at FPR <= 5%.
  The two families without a rule are reported as not decidable by behaviour.

**Outcome (recorded after judging, 2026-11-17).** H14 not supported (error agreement above 0.5965:
TPR 0/20, FPR 0/20). H14b not supported (text decoder TPR 5/5, FPR 0/5; vision CNN TPR 5/5, FPR
5/5). Behaviour therefore stays a reported category only, and is not extended to Hub pairs above
2B parameters.

## Amendment 2026-11-17 (4): H15-H18, follow-ups to every not-supported verdict, before any of their data

Every earlier verdict stands as recorded; nothing below changes or rewords one. H2 and H3 are
measurements of the Hub (its declared lineage is more honest than the hypotheses guessed) and get
no follow-up. H7 was followed up by H13 (supported). Each follow-up starts from a cause found in
data already recorded, and is judged only on data that does not exist yet.

### H15: direction by a combined rule (follows H8)

**Cause, from E2 and E7k.** D1 (dates) was right on 183 of 193 Hub pairs. Of its 10 errors, 9 are
pairs from one organisation released together, where the derived repository (instruct, `-it`,
quantised, fine-tuned) was created 0 to 8 days before its base (Qwen3-14B, gemma-4-31B-it,
Qwen2-VL-2B/7B-Instruct, Qwen2.5-Coder-3B/32B-Instruct, Qwen3.5-35B-A3B, LFM2-24B-A2B-GGUF,
nomic-embed-text-v2-moe); 1 is a base repository re-created years after its children
(runwayml/stable-diffusion-v1-5). D2 (kurtosis) was right on all 35 E7 pairs (100k-400k steps of
pre-training apart) and wrong on 4 of 24 decided fine-tunes: a short fine-tune moves median
kurtosis by less than 0.005 in either direction, and a LoRA merge leaves the sampled tensors'
median gap at exactly 0 (most of the 20 undecided).

**Rule D3 (fixed here; E2's 193 pairs are its dev set).** For a related pair (A, B), not
byte-identical, with Hub creation dates:
1. If A and B have different owners (the part before "/"), or their creation dates are more than
   G = 30 days apart, the earlier is the parent (D1). G is three times the largest gap among D1's
   same-owner dev errors (8 days), rounded up to a month, so it is not fitted to them.
2. Otherwise (same owner, within 30 days) the name decides. Names are compared lower-cased, without
   the owner: if one ends in `-base` and the other equals it without that ending, the `-base` one
   is the parent; else if one is a strict prefix of the other followed by `-` or `_`, the shorter is
   the parent.
3. Otherwise D2 decides if |kurtosis gap| > 0.00062 (D2's dev margin, unchanged).
4. Otherwise the pair is undecided.

D3 on E2 (dev, seen) is reported beside D1 and D2; nothing was tuned beyond the choices above.

**Fresh data.** A new frozen sample (`results/h15_sample.json`, seed 20261118, committed before any
of it is read): 120 frame models per download tier that declare a base and are in neither E2's
nor H13's sample. Each is compared by weights with each declared base exactly as in H13, recording
both creation dates and the kurtosis gap (`experiments/h15_fresh.py`, ledger `H15`). Ground truth:
each declared base the weights confirm (related, not byte-identical) is the parent.

- **H15** On the fresh pairs, D3 is right on >= 95% of the pairs it decides and leaves <= 10%
  undecided. D1 and D2 are reported beside it on the same pairs, with Wilson intervals.

### H16: behaviour calibrated against reference students (follows H9, H12, H14, H14b)

**Cause, from E5e.** Within each training batch the vision-CNN students separate: fresh distilled
dark-knowledge divergence 0.091-0.105 against independent 0.128-0.143; dev 0.164-0.166 against
0.182-0.193. But the whole fresh batch sits below the dev threshold (0.174), so the absolute
threshold admitted all 10. The text decoder shifted the same way (dev distilled 0.42-0.44, fresh
0.345-0.348) and passed only because its gap was wide. An absolute behaviour threshold does not
carry from one batch of students to the next; a comparison with independent students trained the
same way can.

**Score.** For a suspect student S and reference independents R trained with the same recipe on the
same task: drop(S) = (min over R of dark_js(r) - dark_js(S)) / (min over R of dark_js(r)). S is
called *distilled* if drop(S) > r*.

**Rule (dev only).** Dev: every E5e student of the two families whose students learn the task (text
decoder, vision CNN) in both batches already scored (seeds 0-2 and 8-12); each student's references
are its batch's independent students other than itself. r* = the midpoint between the smallest
distilled drop and the largest independent drop, pooled over both families and batches; if that gap
is not positive, H16 is recorded as not supported without a fresh run. `scripts/analyse.py`
computes r* from the ledger.

**Fresh data (GPU; run once leakcheck's Kaggle work releases the quota).** Per family: distilled and
independent students seeds 13-17 (the suspects) and independent references seeds 18-20, trained
exactly as `distill3` (`experiments/distill.py`, same steps), none trained before.

- **H16** On the fresh suspects (10 distilled, 10 independent), drop > r* gives TPR >= 90% and
  FPR <= 5%.

### H16b: the same test where the students learn the task (follows H14b's undecided families)

**Cause, from E5e.** Text-encoder students agree with their teacher on 5-9% of masked positions,
distilled and independent alike, and vision-transformer students on about a third: after 2,000
steps from a random initialisation neither has learned the task, so nothing of the teacher can show
in their behaviour.

**Change.** Text-encoder and vision-transformer students train for 20,000 steps (learning rate 5e-4,
1,000 linear warm-up steps, cosine decay), otherwise as in `distill.py`. A batch counts only if its
independent students reach >= 50% of the teacher's own task accuracy on the probe set (masked-token
top-1; CIFAR-10 top-1), recorded per student; a family whose batch misses this floor is reported as
not decidable, with the accuracies. **Dev batch:** distilled and independent seeds 30-32
(references: the batch's other independents); r*_b chosen exactly as H16's r*. **Fresh batch:**
suspects seeds 33-37 (distilled and independent), references seeds 38-40.

- **H16b** On the fresh suspects of the families that pass the floor, drop > r*_b gives TPR >= 90%
  and FPR <= 5% (pooled over those families; if no family passes the floor, not decidable).

### H17: aligning GGUF tensors when names do not map (follows the unverifiable declared bases)

**Cause, from E2 with E2r, E2g and E2p.** 55 models (58 declared pairs) still declare a base that
cannot be compared by weights: 32 pairs are GGUF files whose architecture the installed llama.cpp
name table does not know, so 0 tensors align (DeepSeek-V4, NeMo Parakeet, newer multimodal models);
10 failed on re-reading (network); 4 use GGUF tensor types the installed `gguf` package does not
list (types 42 and 142); 6 are gated or private (HTTP 401; gated without a byte-identical public
copy); 6 are other formats (a random tiny model with no shared shapes, a draft head, an EXL repack,
an NPU build, no weight files).
*Correction (2026-11-17, same day, before any H15-H18 data): `scripts/analyse.py`'s breakdown
(`reader_fixed.unverifiable`) counts 5 gated or private and 7 other formats, not 6 and 6; the
hand count above put one draft-head pair in the wrong group. No rule depends on these counts.*

**Structural alignment.** When names align fewer than 8 tensors, a GGUF tensor is paired with a
Hugging Face tensor by position: both sides are grouped by block index (`blk.N.` in GGUF; the first
`.layers.N.`, `.layer.N.`, `.h.N.`, `.blocks.N.` or `.block.N.` on the other side; tensors without
an index form one group), and the two must have the same number of blocks; within a group, a tensor
is paired only if its shape occurs exactly once in that group on both sides. Ambiguous shapes are
left out. Nothing else changes (sampling, cosine, thresholds, `MIN_COMPARED`).

**Fork tensor types.** A type is decoded only from a public source: the newest `gguf` release for
mainline types; for a fork's type, that fork's published dequantisation, with a test that our
decoding equals the fork's on the same blocks. A type with no public source is listed with its
repositories and the reason.

- **H17a** On the fresh GGUF pairs where names align >= 8 tensors (H15's GGUF models against their
  declared bases and against the two top-ranked E2-pool candidates of each), structural alignment
  run with names ignored gives the same related / unrelated verdict for >= 98% of pairs (judged
  only if there are at least 20 such pairs, at least 5 of them unrelated by names).
- **H17b** Applied to E2's 32 declared GGUF pairs with no name match (`E2s`, beside E2 as E2g is),
  structural alignment makes >= 50% of them comparable. The 10 network failures are read again in
  the same run. E2 stays as recorded; changed categories are reported beside it.

### H18: behaviour for declared Hub pairs the weights still cannot compare (only if H16 or H16b is supported)

Behaviour on large models proves nothing unless the behaviour check has passed on ground truth, so
H18 runs only if H16 or H16b is supported; otherwise it is recorded as not run, with that reason.
**Scope:** declared pairs still not comparable after H17 whose model and base are both text
language models of <= 14B parameters with the same tokenizer (identical vocabulary). **Score:**
dark-knowledge divergence of next-token distributions on E5's text probes; references: the three
most downloaded frame models with the same tokenizer and a parameter count within 2x that are not
related by weights to the declared base; drop and r* as in H16 (r* from H16's dev rule). A
declaration is *confirmed by behaviour* if drop > r*.

- **H18** The rule confirms >= 90% of the declared pairs it can run (H3 measured 1.3% of comparable
  declarations contradicted, so a valid check should confirm nearly all). Pairs outside the scope
  are counted by reason.

**Outcome of H15 and H17a (recorded after judging, 2026-11-17; the H15 job read all 360 sampled
models, none lost).** Both are **not supported**, and stay so.
- **H15:** D3 is right on 222 of 237 decided fresh pairs (93.7%, Wilson 89.8-96.1%; bar 95%),
  undecided 7 of 244 (bar 10%, met). D1 alone is right on 226 of 240 (94.2%), so on fresh pairs D3 is
  not better than dates alone. Of D3's 15 errors, 9 are pairs with different owners where the derived
  repository is older than its declared base: one repackaging repository (Comfy-Org/Wan-2.1, 5
  pairs) created 50-174 days before the five bases it now declares, and 4 GGUF quantisations created
  0.4-34 days before their base. 5 are within one owner's release where the name rule did not
  apply (`-instruct` vs `-base`, `-MPO`, an L6 model distilled from L12) and kurtosis decided wrongly;
  1 is a checkpoint (`ckpt500`) whose name the rule does not relate to its base. Weight kurtosis
  (D2) is right on 22 of 118 decided fresh Hub pairs: on Hub fine-tunes the child usually has
  the *lower* kurtosis, the opposite of the direction D2 (after MoTHer) assumes, while on pre-training
  checkpoints (E7) it was right on all 35.
- **H17a:** structural alignment agrees with name alignment on 279 of 339 fresh pairs (82.3%; bar
  98%). It never gave the opposite verdict: 271 related and 8 unrelated pairs agree, 0 flip. All 60
  disagreements are pairs it aligned no tensor in (not comparable): 29 models, all Qwen 3.5-3.8
  multimodal or multi-token-prediction GGUF files (`qwen35`, `qwen35moe`) and clip models, where the
  two sides do not have the same number of blocks. H17b (E2's 32 unmatched pairs) is judged on E2s.

**Outcome of H17b (recorded after judging, 2026-11-18; E2s read 895 comparisons, none lost).**
**Not supported:** structural alignment made 11 of the 32 comparable (34%; bar 50%), and stays so.
*Correction to the cause list above:* the "10 failed on re-reading (network)" are not network
failures. Read again by E2s, each declared base is still not readable: it is missing, renamed, or not
a repository id at all (`gpt-oss-20b`, `deepseek-ai/deepseek-r1-14b`); the category is renamed
accordingly. E2s also found two reader gaps: 13 comparisons stopped on GGUF type 42 (E2s started
before that decoder existed) and 3 declared DeepSeek-V4 bases store their FP8 block scales as
safetensors `F8_E8M0` (the OCP Microscaling scale format, 2^(e-127), now decoded). Those 16 are scored
again with E2s's alignment (`e2-struct2`, ledger `E2s`, each pair's latest record counts); H17b is
recomputed on its 32 pairs from those records and the result is reported here as it comes out.
*Result (2026-11-18):* all 16 now compare 24 tensors and are related (the 3 DeepSeek-V4 pairs 0.9994-1.0000);
H17b is 15 of 32 comparable (all related): **not supported**, unchanged. With every re-read, declared
bases the weights confirm are 263 and unverifiable 35; no verdict changes (H2 106/379, H3 3/279, H7 b
88/250).


## Amendment 2026-11-18: H19 and H20, the last follow-ups on direction and alignment, before any of their data

Every earlier verdict stands as recorded. H19 follows H15 and H20 follows H17a/H17b; whatever they
show is recorded, and **no further follow-up on direction or alignment is made after them**. Both
rules below are fixed from causes already recorded (H15's 15 errors, H17a's 60 abstentions); no data
for them has been read.

### H19: direction from when each repository's weights were written (follows H15)

**Cause, from H15.** 9 of D3's 15 errors are repositories *created* before their declared base: a
repackaging repository created months before the bases it now holds, and GGUF quantisations created
days before the base. A repository's creation date is when its name was registered, not when its
weights arrived; the weights of a derivative cannot have been written before its parent's.

**Rule D4.** For a repository at the commit read, its *content date* is the earliest last-commit
date among its current weight files (`.safetensors`, `.gguf`, `.bin`, `.pt`, `.pth`, `.ckpt`; the
Hub's `tree?expand=true` listing at that commit). Taking the earliest file makes a later conversion
added to an old repository (a `model.safetensors` beside a 2020 `pytorch_model.bin`) leave the date
unchanged. Of a related pair, the one with the earlier content date is the parent; the pair is
*undecided* when the two dates are equal to the second or either is missing.

**Fresh data.** A new frozen sample (`results/h19_sample.json`, seed 20261119, committed before any of
it is read): 120 frame models per download tier that declare a base and are in none of E2's, H13's or
H15's samples; where a tier has fewer than 120 such models left (the top 1k, after three samples), all
of them. (Added the same day, when drawing showed the top tier short, before any model was read.) Each is compared by weights with each declared base exactly as in H15, and both
repositories' content dates are recorded (`experiments/h19_fresh.py`, ledger `H19`). Ground truth: each
declared base the weights confirm (related, not byte-identical) is the parent.

- **H19** On the fresh pairs, D4 is right on >= 95% of the pairs it decides and leaves <= 10%
  undecided. D1 (creation dates) and D3 are reported beside it on the same pairs, with Wilson intervals.

### H20: structural alignment across unequal block counts (follows H17a and H17b)

**Cause, from H17a.** Structural alignment never gave a wrong verdict but aligned nothing in 60 of 339
pairs, all multimodal or multi-token-prediction checkpoints and clip models. Its rule required both
sides to have the same number of blocks, counted over *every* indexed tensor: a Hugging Face
checkpoint that also holds a vision tower (`visual.blocks.N`) or a multi-token-prediction layer
(`mtp.layers.0`), or a GGUF file that appends extra blocks, fails that test even though its language
model's blocks correspond one to one; and clip GGUF files name their blocks `v.blk.N` / `t.blk.N`,
which the rule did not read as blocks.

**Structural alignment v2.** Each side's indexed tensors are split into *stacks* by the name part
before the index (`blk`, `v.blk`, `t.blk` in GGUF; e.g. `model.language_model.layers`,
`model.visual.blocks`, `mtp.layers` on the other side). GGUF stacks are taken largest first; each is
paired with the unpaired stack of the other side with the same number of blocks, else the one with
the nearest number of blocks (ties: more tensors). In a stack pair, block i is aligned with block i
for i below the smaller of the two counts; blocks beyond it are left out. Tensors with no index form
one group on each side. Within a group, a tensor is paired only if its shape occurs exactly once in
that group on both sides, as in v1. Nothing else changes (sampling, cosine, thresholds, `MIN_COMPARED`).

- **H20** On the fresh GGUF pairs where names align >= 8 tensors (H19's GGUF models against their
  declared bases and against the two top-ranked E2-pool candidates of each), v2 run with names ignored
  gives the same related / unrelated verdict for >= 98% of pairs (judged only if there are at least 20
  such pairs, at least 5 of them unrelated by names). Opposite verdicts and pairs v2 aligns too few
  tensors in are reported separately, and v1 is reported beside it on the same pairs. Applied to the 14
  declared E2 GGUF pairs neither names nor v1 align, the number v2 makes comparable is reported (a
  measurement, no bar).

**Outcome of H19 and H20 (recorded after judging, 2026-11-18; Kaggle job `pedigree-h19` read all 355
sampled models, none lost; the 14 E2 pairs, ledger `H20e2`, all read).**

- **H19: not supported.** D4 is right on 236 of 263 decided pairs (89.7%, Wilson 85.5-92.9%) and leaves
  0 of 263 undecided. On the same pairs D1 is right on 236/260 and D3 on 244/257. Of D4's 27 errors,
  10 are one multi-model repackaging repository (Comfy-Org) whose earliest weight file belongs to a
  different model than the declared base; 10 are same-owner releases uploaded within 2.2 days of each
  other (base and instruct or GGUF; in four, the derived files were committed seconds or minutes
  before the base's); and in 7 the base's current weight files were committed 0-77 days after the
  derivative's.
- **H20: not supported.** v2 agrees with names on 373 of 382 pairs (97.6%; 15 of them unrelated by
  names, so the test was judged). v1 agrees on 331 of the same 382. One opposite verdict:
  `city96/umt5-xxl-encoder-gguf` vs `google/umt5-xxl`, related by names (0.99999) and unrelated by v2
  (0.0024), because the base's `encoder.block` and `decoder.block` stacks have the same block count and
  the tie-break (more tensors) chose the decoder. 8 pairs v2 aligned 0-1 tensors in, all upscaled or
  merged models. On E2's 14 frozen pairs, v2 made 2 comparable (>= 8 tensors); 12 score above a
  threshold, 10 of them on 3-7 tensors, which is below `MIN_COMPARED`, so they are not counted as
  comparable.
- As set out in this amendment, no further follow-up on direction or alignment is made. The audit
  keeps D1 dates for direction and names-then-v1 for GGUF alignment. Open: H16 and H16b (GPU), and H18
  only if one of them is supported.

## Amendment 2026-11-20: closing every remaining gap, before any of its data

The project owner asked on 2026-11-20 that no gap remain that work could close. This amendment
re-opens alignment (closed by the 2026-11-18 amendment) on the owner's instruction; direction stays
closed (H8, H15 and H19 tried every rule the data suggested; the audit reports D1 with its measured
accuracy). Every earlier verdict stands as recorded. E2 stays as recorded; every new reading is
reported beside it, as E2r, E2g, E2p, E2f and E2s were. All rules below are fixed before any of the
models they apply to is read with them; the readers were checked only on repositories outside every
sample, and against reference libraries (tests).

### R1: readers for every weight format with a public specification

**Cause.** 46 of E2's 600 models were never read (E2 counted them "not readable"), so they were
never searched for a parent or compared with a declared base, and every re-reading since (E2r, E2g,
E2p, E2f, E2s) re-scored *pairs* only. Their reasons: a pre-zip PyTorch file (12; the reader has
decoded this format since E2p, but these models were never read again), ONNX only (10), no weight
file in a known place (10: CTranslate2 `model.bin`, TFLite, LiteRT-LM, and repositories with no
weights), other checkpoint formats (6: `.pt`, `.pth`, `.ckpt`, `.nemo`, MLX `.npz`), gated (3), a
flair pickle the restricted unpickler refused (1), HTTP 403 (1), and GGUF repositories whose files
sit in subfolders (3).

**Readers** (`pedigree/formats.py`, `pedigree/torchzip.py`), each tested against the format's
reference library on the same file (`tests/test_formats.py`): ONNX (initializers, Constant nodes,
external data; `DequantizeLinear` weights read as values), NumPy `.npz`, CTranslate2 `model.bin`
(int8 weights read as values with their row scales), NeMo `.nemo` (the PyTorch checkpoint inside
the tar, read in place), PyTorch `.pt`/`.pth`/`.ckpt` (optimizer and discriminator states left
out), TFLite (flatbuffer schema; int8 weights read with their scales), PaddlePaddle `.pdiparams`,
and LiteRT-LM if its container schema is public (otherwise listed with the source checked). The
restricted unpickler now stands in an inert placeholder for an unknown *class* (keeping its state,
so tensors inside a framework's objects are found) and still refuses every function and every
torch/numpy global other than the tensor records; no code from any file runs. GGUF shard sets in
subfolders are chosen by the same precision preference as top-level ones.

**E2x.** Every E2 model recorded "not readable", and every declared or pool pair whose other side
was unreadable, is read again with R1 and audited exactly as E2 audited a model (declared bases,
the E2 pool, the invariant score, the same thresholds), ledger `E2x`. `scripts/analyse.py`
reports E2 as recorded and, beside it, E2 with E2x applied.

- **H21** After E2x, no E2 model is unreadable because of its file format, except formats with no
  public specification (each listed with the source that was checked). Models unreadable because
  the repository holds no weights, or is gated or private, are counted by reason and do not count
  against H21.

### R2: what a declaration names, for the pairs weights still cannot compare

Every declared pair that is not comparable after E2x and R3 is put in exactly one category, by these
rules in order, so no pair is left as an unexplained "unverifiable": (1) the declared base is not of
the form `owner/name` ("not a repository id"); (2) the Hub redirects it to another repository
(renamed): the pair is read again under the new name; (3) the Hub answers 401 or 404 to an anonymous
request ("not found on the Hub, or private": the Hub does not tell the two apart); (4) gated (the
repository exists and asks for terms; read when an account that accepted them is available); (5)
model and base share fewer than 8 tensors of equal shape ("different structure": a draft head, a
random-weight model of the base's architecture, a model that uses the base as one part); (6) a weight
format with no public specification. A pair in none of these is listed by name as a gap of the audit.

### R3 and H23: alignment v3 (re-opened 2026-11-20)

**Cause, from H20.** v2 gave one opposite verdict: an encoder-only GGUF paired with its base's decoder
stack, because the two stacks have the same block count and the tie-break preferred the larger. It
aligned 0-1 tensors in 8 upscaled or merged models whose block counts differ from their base's. And 14
E2 GGUF pairs align under no rule: their GGUF files keep their own tensor names (ComfyUI-GGUF
diffusion models, whisper.cpp, parakeet and TTS converters, and new architectures).

**Alignment v3** (`pedigree/structalign.py`), used whenever names pair fewer than 8 tensors, for any
two formats (not only GGUF): (a) *direct names*: a tensor whose name, after removing a leading
`model.`, `model.diffusion_model.`, `diffusion_model.` or `transformer.`, equals a name on the other
side with the same shape (the GGUF shape reversed) is paired; (b) *role-paired stacks*: stacks are
paired first by a shared role word in their prefix (`encoder`/`enc`, `decoder`/`dec`, `vision`/
`visual`/`v`, `text`/`t`, `audio`/`a`, `language_model`), then as in v2; (c) *unequal depth*: in a
stack pair with n and m blocks (n != m), block i is aligned with block round(i * (m - 1) / (n - 1)), an
order-preserving map that uses no weight values; within each block pair, tensors are paired by unique
shape as in v1. (a) is tried first, then (b) with (c); the mapping that pairs more tensors is used. *(Added the same day, before any model was read with v3:)* (d) a 2-D tensor whose shape occurs once on each side only when reversed (ONNX and TFLite store linear weights transposed) is paired with its transpose, read whole (tensors over 64 MB are left out). v3 is applied when names pair fewer than 8 tensors (`auto3`); H23a runs it with names ignored (`struct3`).

**Fresh data.** A new frozen sample (`results/h23_sample.json`, seed 20261120, drawn before any of it
is read): frame models in none of E2's, H13's, H15's or H19's samples that declare a base and whose
weights are GGUF, ONNX, CTranslate2 or NeMo, up to 150 per download tier (all where fewer remain). *(Same day, before drawing: the format is taken from the frame's `library` field, GGUF = gguf, GGUF, llama.cpp, llama-cpp, llamacpp, llama_cpp, llama-cpp-python, node-llama-cpp, ggml, ik_llama.cpp, comfyui-gguf-reboot, wllama, ollama; ONNX = onnx, onnxruntime, onnxruntime-web, onnxruntime-genai, transformers.js, gliner2-onnx, onnx-asr; CTranslate2 = ctranslate2, faster-whisper; NeMo = nemo. The format actually read is recorded per model.)*
Each is aligned with each declared base and the two top-ranked E2-pool candidates under names, v2
and v3.

- **H23a** On the fresh pairs where names align >= 8 tensors, v3 run with names ignored gives the
  same related / unrelated verdict for >= 98% and the opposite verdict for none (judged if >= 20
  pairs, >= 5 of them unrelated by names).
- **H23b** On the fresh declared pairs where names align fewer than 8 tensors, v3 makes >= 80%
  comparable (>= 8 tensors). The 14 E2 pairs, on which v3 was designed, are reported beside it and do
  not count.

### H24: behaviour for speech and diffusion models (GPU, with H16/H16b)

**Cause.** E5, H16 and H16b test behaviour on four families; speech and diffusion have none.

**Design.** As H16b: half-depth students from a random initialisation, distilled (imitating the
teacher's outputs) and independent (trained on the task's own targets), 20,000 steps (learning rate
5e-4, 1,000 warm-up steps, cosine decay). *Speech:* teacher facebook/wav2vec2-base-960h; distilled
students imitate its per-frame CTC distributions (KL, temperature 2) on LibriSpeech train-clean-100
audio, independent students train with CTC on its transcripts; score: dark-knowledge divergence of
per-frame distributions on 200 LibriSpeech test-clean utterances. *Diffusion:* teacher
google/ddpm-cifar10-32 for both batches; distilled students regress the teacher's noise prediction,
independent students the true noise, on CIFAR-10 train; score: mean squared difference from the
teacher's noise prediction on 1,000 CIFAR-10 test images at 10 fixed timesteps, divided by the
teacher's own error (lower = closer). Competence floor: a family counts only if its independent
students reach >= 50% of the teacher's task score (speech: 1 - character error rate; diffusion:
1 - normalised noise error), recorded per student. Dev batch seeds 50-52 (distilled and independent);
fresh suspects 53-57, references 58-60; r*_c chosen on dev exactly as H16's r*.

- **H24** On the fresh suspects of the families that pass the floor, drop > r*_c gives TPR >= 90% and
  FPR <= 5% (pooled; not decidable if no family passes the floor).

H18 runs if H16 or H16b is supported (only they test text language models; H24 does not open H18).

**Outcome of H23 (recorded after judging, 2026-11-20; Kaggle job `pedigree-h23` read all 300 sampled
models, none lost).**

- **H23a: not supported.** v3 (names ignored) agrees with names on 518 of 526 pairs, 37 of them
  unrelated by names (judged); 6 abstain (v3 aligns too few) and 2 are opposite. Both opposite verdicts
  are one model, `SalahALHaismawi/Nuqta-Arabic-OCR-4B-GGUF`, against two Qwen3.5-4B pool candidates:
  related by names (median 0.507), unrelated by v3 (0.004). The GGUF has 35 blocks and the base 32;
  read by name, its blocks 0-14 equal the base's blocks 0-14 (cosine >= 0.996) and its later blocks do
  not, so the model keeps the base's blocks at their own indices, and v3's proportional map (c) paired
  block i with round(i * 31 / 34), off by one from block 6 on.
- **H23b: not supported.** v3 makes 73 of 145 fresh declared pairs comparable. Of the 72 it does not:
  30 are ONNX exports (transformers.js and others) whose large weights are initializers with no name
  (`onnx::MatMul_N`), stored transposed, so only biases and norms pair; 36 pair fewer than 8 large
  tensors because tensors of one shape repeat inside a block (attention q/k/v/o, two feed-forward
  layers), which (c)'s unique-shape rule leaves unpaired, or a converter dropped a size-1 dimension
  (`(2048, 1024)` against `(2048, 1024, 1)`), or the two layouts really differ (fused and split
  attention); 6 could not be read: GGUF type 41 (Q1_0, which the gguf package lists but cannot decode;
  4 others use type 103, which no public source defines).

## Amendment 2026-11-20 (2): H25, alignment v4, after H23 and before any of its data

The project owner's instruction of 2026-11-20 (no gap left that work can close) stands. H23's verdicts
stand as recorded; H23's sample is design data for v4 and is reported beside H25, not counted.

**Readers.** (i) GGUF type 41 (Q1_0) is decoded from llama.cpp's `dequantize_row_q1_0` (weight j is
bit j % 8 of byte j / 8: +d if set, else -d), with a parity test against that C function. (ii) ONNX:
an initializer whose name begins `onnx::` and which is input 1 of exactly one `MatMul` node, or of a
`Gemm` node, takes that node's path as its name (`/encoder/layer.0/attention/query/MatMul` ->
`encoder.layer.0.attention.query.weight`). `MatMul` (and `Gemm` without `transB`) stores the weight as
[in, out], so it is seen as its transpose [out, in] (read whole; over 64 MB it keeps its anonymous name).
E2x and H23 are reproduced with these off (`PEDIGREE_READERS_V4=0`).

**Alignment v4** (`pedigree/structalign.py`): v3 with three changes, none using weight values except (c4).
(e) Shapes are equal when they are equal after removing size-1 dimensions (same element order).
(f) Tensors of one shape that repeat within a block pair (or among tensors with no block index) are
paired by name: the leaf names (the part after the block index; lower case, only letters and digits)
are compared by longest-common-subsequence length, and a and b are paired when each is the other's
single best match, strictly above its second best.
(c4) In a stack pair with n != m blocks, three order-preserving block maps are built: equal index from
the first block (i -> i), equal index from the last block (i -> i + m - n), and v3's proportional map.
Each is scored as usual and the one with the highest median plain score is kept (the map is recorded).
With n = m, only i -> i is used.
v4 is applied when names pair fewer than 8 tensors with >= 4,096 values each (`auto4`; v3 counted
tensors of any size). H25a runs it with names ignored (`struct4`).

**Fresh data.** A new frozen sample (`results/h25_sample.json`, seed 20261121, drawn before any of it
is read): the H23 rule (same library field values, a declared base, up to 150 per download tier),
excluding E2's, H13's, H15's, H19's and H23's samples. Each model is aligned with each declared base
and the two top-ranked pool candidates under names, v3 and v4.

- **H25a** On the fresh pairs where names align >= 8 tensors, v4 with names ignored gives the same
  related / unrelated verdict for >= 98% and the opposite verdict for none (judged if >= 20 pairs, >= 5
  of them unrelated by names). Picking the best of three block maps can only raise a score; an unrelated
  pair turning related counts as opposite.
- **H25b** On the fresh declared pairs where names align fewer than 8 tensors and both models were
  read, v4 makes >= 80% comparable (>= 8 tensors). Pairs that could not be read count under H21 and R2,
  not here. H23's 145 pairs are reported beside it under v4 and do not count.

**Outcome of H21 (recorded after judging, 2026-11-21; Kaggle job `pedigree-e2x` read all 46 models,
225 pairs and 49 v3 pairs, none lost).** **Not supported.** 37 of the 46 models E2 never read are now
read. Four are still unreadable for reasons the readers could have handled, so H21 fails as stated:
three GGUF repositories (`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`,
`AhmedPPL/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF`, `unsloth/Inkling-Small-GGUF`) keep every
quantisation in a subfolder and one multimodal projector (`mmproj-*.gguf`) at the top level; the
top-level rule took the projector, the GGUF chooser then set it aside as not the model, and the
subfolders were never looked at. The fourth, `flair/chunk-english`, pickles framework records the
restricted unpickler refused: `collections.defaultdict`, then `torch.torch_version.TorchVersion`,
then the class `torch.optim.sgd.SGD`. The other five are counted by reason, as H21 allows: no weight
files 3, gated with no public copy 1, HTTP 403 1. With E2x applied, R2 puts 24 of the 30 declared
pairs still not comparable into a category; 6 are listed as gaps (`Comfy-Org/Qwen3-VL`,
`tiny-random/qwen3.5-moe`, `zgpu-ai/zlm-v1-signal-extract`, `z-lab/Qwen3.6-35B-A3B-DFlash`,
`KoreaPeter/ms-eff-gcvit-deepfake-b5-celeb-df-v2`, `onnx-community/Nemotron-3-Diarization-ONNX`).

**Outcome of H25 (recorded after judging, 2026-11-21; Kaggle job `pedigree-h25` read all 300 sampled
models and H23's 147 design pairs, none lost).**

- **H25a: not supported.** v4 (names ignored) agrees with names on 602 of 604 pairs, 61 of them
  unrelated by names (judged); one abstains and one is opposite. The opposite verdict is
  `onnx-community/LFM2-700M-ONNX` against `LiquidAI/LFM2-700M`: related by names (11 tensors, median
  1.0), unrelated by v4 (24 tensors, 0.0099). This exporter names its matrix weights
  `<module>.MatMul.weight` and stores them as [in, out]; reader (ii) turned only anonymous (`onnx::`)
  initializers round, so v4 paired them by shape with the wrong partners (`mlp.gate_proj` [1536, 6912]
  with the base's `feed_forward.w2`, every square matrix transposed). The abstention is
  `LiquidAI/d1-omni-600M-GGUF` (names: 24 tensors, 0.99998; v4: 1): its base is multimodal, and v3's
  stack rule (nearest block count, then size) paired the GGUF's text stack with the base's audio
  encoder, which shares only a few bias shapes with it.
- **H25b: not supported.** v4 makes 67 of 99 fresh declared pairs comparable (2 more could not be
  read). Read again with the readers below as a diagnosis (not a ledger run), the 32 it does not make
  comparable are: ONNX encoder-decoder exports read as their largest file only, a merged decoder whose
  nodes sit in `If` branches (8: Whisper x4, opus-mt x3, TrOCR); ONNX with onnxruntime's dynamic
  quantisation, `MatMulInteger` (2) or 4-bit `MatMulNBits` (1); a side file taken for the model (8: a
  multi-token-prediction head, `dspark`/`DSpark` draft models, a VAE, a classifier head, a control
  vector, side `.safetensors` files and an `Engram` table, each with far fewer tensors than the model
  beside it); a converter that numbered its tensors (`t0000`...) (1); a base held under a leading path
  with LoRA unmerged (1); and 11 others: GGUF architectures llama.cpp's table does not know (`ltx2`
  x2, `qwen35moe` with fused experts), different structures (a prefill draft model, `odp`), a fork's
  undocumented GGUF type 103, and ONNX or PyTorch pairs that align 2-7 large tensors. H23's pairs
  under v4 (design data): 110 of 142 declared pairs comparable, both former opposite verdicts agree.

## Amendment 2026-11-21: readers v5, alignment (g)-(i), E2y and H26, after H21 and H25 and before any of H26's data

H21's and H25's verdicts stand as recorded. H25's pairs are design data for v5 and are reported beside
H26, not counted. The project owner's instruction of 2026-11-20 stands.

**Readers v5** (`pedigree/formats.py`, `pedigree/torchzip.py`, `pedigree/units.py`), each tested on a
file its reference library wrote, the values read checked against the values written
(`tests/test_readers_v5.py`): (iii) an encoder-decoder ONNX export (`encoder_model*.onnx` beside
`decoder_model*.onnx`) is read as both files of one precision, the decoder without past keys and not
merged where that file exists; the encoder's names take `encoder.` unless they begin with `encoder`
or `model.`. (iv) Nodes inside sub-graphs (`If`, `Loop`, `Scan`) count as uses, and a merged
decoder's `_merged_N` suffix is dropped. (v) Every initializer that is input 1 of exactly one
`MatMul`, `MatMulInteger` or `Gemm` without `transB` is seen as its transpose, whatever its name (the
operators define it as [in, out]); an anonymous one takes the node's path, and a `.MatMul` component
before `.weight` is dropped. (vi) onnxruntime's dynamic quantisation (`X_quantized` with `X_scale` and
`X_zero_point`) is read as the values it stands for, named X. (vii) When a repository holds several
weight units (a top-level safetensors or PyTorch set and GGUF sets, or GGUF sets of different files),
the one whose headers list the most tensors is the model; units with equally many (one model's
quantisation variants) are chosen among as before, and a multimodal projector never counts. (viii)
onnxruntime's `MatMulNBits` (4- and 8-bit blocks, one scale per block, zero points packed per row,
default 2^(bits-1)) is read as [N, K] values. (vi) and (viii) are checked against onnxruntime itself
(1.31.1, and 1.23.2 on Python 3.10, for which 1.31 publishes no wheel): the weight it runs is read
from its output on an identity input. The restricted unpickler stands in a plain mapping for
`collections.defaultdict` (its factory, a plain value type, is never called), `str` for
`TorchVersion`, a tuple for `torch.Size`, and an inert placeholder for any other torch *class* (an
optimizer type); torch and numpy functions and every other function stay refused, an optimizer's
state is not taken for weights, and no code from any file runs. `pedigree verify` and the browser
checker choose a repository's files by the audit's rule. `PEDIGREE_READERS_V5=0` reproduces E2x and
H25.

**Alignment v4 with (g)-(i)** (`pedigree/structalign.py`, with readers v5; `PEDIGREE_READERS_V5=0`
reproduces H25): (g) when one side's names are numbers only, its tensors in number order pair by
position with the other side's, in header order and in name order, each order kept only if the two
shape sequences are equal throughout (size-1 dimensions aside); where both qualify, each is a
candidate, scored like (c4)'s maps. (h) When one model holds the other under a leading path, the path
most often left when a name is cut down to one on the other side is removed, PEFT's `.base_layer` is
read as the weight, and names then equal with equal shapes pair. (g) and (h) compete with direct names
by the number of large tensors they pair. (i) A stack's partner is the stack whose shapes overlap its
own most (multiset intersection), then v3's rule.

**E2y.** Every E2 model still unreadable after E2x, every pair whose other side is still unreadable,
and every declared pair still not comparable (`results/e2y_targets.json`, frozen from the ledger
before any reading) is read again with readers v5 and aligned by `auto4`, ledgers `E2y`, `E2yp`,
`E2v4`. `scripts/analyse.py` reports E2 with E2x and E2y applied beside E2 as recorded, and R2 on it.

**Fresh data.** A new frozen sample (`results/h26_sample.json`, seed 20261122, drawn after this
amendment is committed and before any of it is read): H25's rule, excluding E2's, H13's, H15's,
H19's, H23's and H25's samples. Each model is read and aligned exactly as in H25.

- **H26a** On the fresh pairs where names align >= 8 tensors, v4 with (g)-(i) and names ignored gives
  the same related / unrelated verdict for >= 98% and the opposite verdict for none (judged if >= 20
  pairs, >= 5 of them unrelated by names).
- **H26b** On the fresh declared pairs where names align fewer than 8 tensors and both models were
  read, `auto4` makes >= 80% comparable (>= 8 tensors).
- **H26c** (follows H21) No fresh sampled model, and no declared base of one, is unreadable because of
  its file format, except formats with no public specification (each listed with the source checked);
  no weight files, gated, private or missing repositories are counted by reason and do not count.

**Running H26 and E2y (2026-11-22, before any verdict).** The first Kaggle sessions (`pedigree-h26`,
`pedigree-e2y`) were stopped at Kaggle's 12-hour limit. `pedigree-h26` had read 150 of the 300 fresh
models, losing the other 150 to its time budget; it never reached the design pairs. `pedigree-e2y`
had read the 9 models and 70 of the unread pairs. Two causes were found in the logs and fixed
without changing what is read or how it is judged:
(1) Rule (vii) counted each GGUF unit's tensors by walking its whole metadata, vocabularies of
several MB included, about 20 minutes for a repository of 9 units. It now reads the count from each
file's first 24 bytes and sums a split set's shards, which by the format's definition equals
`split.tensors.count` (`tests/test_readers_v5.py`).
(2) Two threads could read a Hub metadata or config cache file while a third was still writing it
(`JSONDecodeError`: one model and one pair of E2y). Those files are now written whole
(write, then rename), and a file cut short is fetched again.
The fresh sample resumes from the 150 records already written (`results/kaggle/h26/h26_raw.jsonl`).
Its rule and readers are the same, so the 150 are not read again. The design pairs and E2y run from
the start, each in a session of its own.

**Outcome of H16 (recorded after judging, 2026-11-22; the fresh students were trained on friends'
GPU accounts, `distill4-*`, and scored by `pedigree-e5e-h16-*`, 16 students per family, none
failed).** **Supported.** The dev rule, fixed before the fresh students existed, is drop > 0.063
(dev gap 0.057). On the fresh suspects all 10 distilled students exceed it (drops 0.150 to 0.349)
and none of the 10 independent students do (drops -0.094 to -0.002): TPR 10/10, FPR 0/10, in both
families (text decoder 5/5 and 0/5, vision CNN 5/5 and 0/5). This is the first behaviour rule
in the project to pass on fresh students. It opens H18.

**Outcome of H18 (recorded 2026-11-22, after H16 was judged; `experiments/h18_scope.py`, before any
behaviour was run).** **Not judged.** Its scope, as written above, is declared pairs still not
comparable after H17 whose model and base are both text language models of <= 14B parameters with
one tokenizer. The scope step puts each of the 42 such pairs in one category, rules in order
(`results/h18_scope.json`). The counts: the model or base is not readable 18 (15 of them gated or
HTTP 401, the project owner having decided on 2026-11-21 not to read gated repositories; 2 with no
weight files; 1 base in another checkpoint format, `nvidia/parakeet-tdt-0.6b-v2`), not a
text language model 13 (vision-language, speech, TTS, image and video models), more than 14B
parameters 10, runs only through the repository's own code 1
(`mobilint/Qwen2.5-0.5B-Instruct-Batch16`: an `auto_map` to `proxy_qwen2.py` and an NPU binary;
Pedigree runs no code from a repository). No pair is in scope, so the rule has nothing to confirm;
H18 is recorded as not judged, not as passed.

**Outcome of H26 (recorded after judging, 2026-11-22; Kaggle `pedigree-h26` read all 300 sampled
models in two sessions, none lost).**

- **H26a: not supported.** v4 with (g)-(i), names ignored, agrees with names on 598 of 599 pairs, 46
  of them unrelated by names (judged); none abstain and one is opposite:
  `onnx-community/siglip2-so400m-patch14-384-ONNX` against `google/siglip2-so400m-patch14-384`
  (names: 24 tensors, median 1.0; v4: 24, 0.005). The export holds the vision tower only; the base
  holds a vision and a text tower of the same depth and the same shapes. Rule (i) scored both towers
  equal (432 shared tensor shapes each), and so did every later tie-break (depth, size). The partner
  was then the first of a Python set, an order that changes between runs with the hash seed. On
  Kaggle it was the text tower. The same code run locally paired the vision tower.
- **H26b: supported.** auto4 makes 90 of the 108 fresh declared pairs that names align fewer than 8
  tensors in comparable (83%); 1 more declared pair could not be read (counted under H26c).
- **H26c: supported.** 15 repositories (sampled models or their declared bases) are unread, none
  because of a file format: HTTP 401 8, HTTP 403 1, gated with no byte-identical public copy 3, no
  weight files 3.

**Amendment 2026-11-22: stack partners are chosen the same way in every run (after H26, before any
later data).** In v2, v3 and v4 alike, a stack's partner was the minimum over a set by depth (and in
v4 shape overlap) and size. When all of these tie, Python's set order, which varies between runs
with the hash seed, chose the partner. A tie now goes to the stack whose name shares the longest
subsequence with the other side's stack name, then to name order (`pedigree/structalign.py` `_tie`;
`tests/test_structalign.py` checks one result under eight hash seeds, and fails without the fix).
Read again with it, under two hash seeds, H26a's opposite pair aligns 24 tensors at median 1.0
(related) both times. H26a's verdict stands as recorded; the pair is design data for this rule.

## Amendment 2026-11-22 (2): rule (j) and H27, after H26 and before any of H27's data

H26's verdicts stand as recorded. Its pairs are design data for what follows, reported beside H27
and not counted.

**Cause.** H26d's one disagreement is `Xenova/vit-gpt2-image-captioning` against
`nlpconnect/vit-gpt2-image-captioning`, an ONNX export of the same model: names gave unrelated (24
tensors, median 0.005), v4 related (1.0). Its decoder is GPT-2, whose Conv1D layers store
[in, out], which is exactly what an ONNX MatMul holds. Rule (v) took them as a Linear's [out, in]
and transposed them, so the non-square ones no longer matched the base by shape. The square ones
(768 x 768: `attn.c_proj`, `crossattention.q_attn`) paired by name and were compared transposed, at
cosines near 0.

**(j)** (`pedigree/verify.py` `orient`, applied to every comparison under readers v5). For each
side with tensors (v) transposed: among the non-square ones whose name the other side holds, if
more match the other side's shape untransposed than transposed, and at least 2 do, all of that
side's (v) tensors are read untransposed. `tests/test_readers_v5.py` checks it on a Conv1D-layout
and a Linear-layout ONNX file against their safetensors bases: all four weights compare at 1.0
under names and auto4, and the Conv1D case fails without (j).

**Fresh data.** A new frozen sample (`results/h27_sample.json`, seed 20261123, drawn after this
amendment is committed and before any of it is read): H25's rule, excluding E2's, H13's, H15's,
H19's, H23's, H25's and H26's samples. Each model is read and aligned exactly as in H26, with
(j) and the tie-break of the first 2026-11-22 amendment. Design data (`H27d`): H26's pair with the
opposite verdict, H26d's disagreement, and H26's 18 declared pairs auto4 left not comparable.

- **H27a** On the fresh pairs where names align >= 8 tensors, v4 with (g)-(i), the tie-break and
  (j), names ignored, gives the same related / unrelated verdict for >= 98% and the opposite
  verdict for none (judged if >= 20 pairs, >= 5 of them unrelated by names).
- **H27b** On the fresh declared pairs where names align fewer than 8 tensors and both models were
  read, `auto4` makes >= 80% comparable (>= 8 tensors).
- **H27c** No fresh sampled model, and no declared base of one, is unreadable because of its file
  format, except formats with no public specification; no weight files, gated, private or missing
  repositories are counted by reason and do not count.

**Outcome of H27 (recorded after judging, 2026-11-22; Kaggle `pedigree-h27` read all 217 sampled
models in one session, none lost).**

- **H27a: not supported.** v4 with (g)-(i), the tie-break and (j), names ignored, agrees with names
  on 406 of 408 pairs (34 unrelated by names); none abstain and two are opposite, both GGUF files of
  GPT-2 models against their Hugging Face base: `opencerebral/Boris-250M-Instruct-GGUF` against
  `KSP-NMAI/Boris-250M-Instruct` and `Attoloop/tensor-diet-bf16-gguf` against
  `openai-community/gpt2-large` (names related, v4 unrelated at 19 tensors, medians 0.02 and 0.02).
  llama.cpp writes GPT-2's Conv1D weights in a Linear's layout, [out, in]; the Hugging Face file keeps
  [in, out]. Names ignored, a block's MLP up projection then has the base's down projection's shape,
  and v4 paired them (every block, both pairs); the square attention outputs compared transposed.
- **H27b: not supported.** auto4 makes 53 of the 83 fresh declared pairs that names align fewer than
  8 tensors in comparable (64%); 5 more declared pairs could not be read (counted under H27c). Of the
  30 left, 21 share fewer than 8 tensors of >= 4096 values by shape in either orientation, so no
  alignment can compare them (13 are one GPT-2 GGUF that declares 14 bases of other architectures).
  Of the 9 that share more: 4 were read from the wrong file (below); 3 are multi-token-prediction or
  draft heads saved alone, against the whole model; 1 uses a GGUF tensor type llama.cpp does not
  publish; and 1, `onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX`, comes from PyTorch's newer
  ONNX exporter, which names weights `val_N` and its nodes `node_MatMul_N` with no module scope, so a
  layer's four 384 x 384 attention weights cannot be told apart by name or shape (v4 abstains).
- **H27c: supported.** 9 repositories are unread, none because of a file format: HTTP 401 2, gated
  with no byte-identical public copy 4, no weight files 3.

## Amendment 2026-11-22 (3): reader (ix), rule (k) and H28, after H27 and before any of H28's data

H27's verdicts stand as recorded. Its pairs are design data for what follows, reported beside H28
and not counted.

**Cause (reader).** In `turintech/Qwen3.5-4B-Artemis-GGUF`, `W4D/Nemotron-3.5-Embed-1B-GGUF`,
`csdhall/Nanbeige4.3-33B-GGUF` and `yanolja/YanoljaNEXT-Rosetta-27B-2511-GGUF` the file read was
llama.cpp's importance matrix (`*imatrix*.gguf`: per-weight activation statistics used to quantise,
`.in_sum2` and `.counts`, no weights). It lists more tensors than the model, so (vii) chose it.

**(ix)** (`pedigree/units.py`, readers v6; `PEDIGREE_READERS_V6=0` reproduces H27). A `.gguf` file
named "imatrix" with no quantisation tag whose header's `general.type` is "imatrix" is set aside
before any other rule; in (vii), a GGUF unit whose header says so counts no tensors. Read again with
it, all four repositories open a weight file (`Qwen3.5-4B-Artemis-Q5_K_M.gguf` against
`Qwen/Qwen3.5-4B`: 24 tensors at median 0.999 under names, struct4 and auto4).

**(k)** (`pedigree/verify.py` `orient_gguf`, applied to every comparison under readers v6). When
exactly one side is GGUF: of the other side's non-square 2-D tensors, count those whose shape the
GGUF side holds as stored and those it holds reversed. If reversed wins, and at least 2 are, that
side's square 2-D tensors and its non-square ones held reversed are read transposed. Shapes alone
decide, so (k) is the same with names used or ignored. Read again with it, both of H27a's opposite
pairs align 24 tensors at median 1.0 under names, struct4 and auto4.

`tests/test_readers_v5.py` checks (ix) (an imatrix beside a model, a model named "imatrix", a file
with an imatrix header and another name) and (k) (a GPT-2 GGUF against a Conv1D-layout and a
Linear-layout base: all eight weights at 1.0 under names, struct4 and auto4); the imatrix case and
the Conv1D case fail without them.

**Fresh data.** A new frozen sample (`results/h28_sample.json`, seed 20261124, drawn after this
amendment is committed and before any of it is read): H25's rule, excluding E2's, H13's, H15's,
H19's, H23's, H25's, H26's and H27's samples. Each model is read and aligned exactly as in H27,
with (ix) and (k); each declared pair also records how many tensors of >= 4096 values the two hold
of the same shape in either orientation. Design data (ledger `H28-design`): H27a's two opposite
pairs and H27b's 30 pairs left not comparable.

- **H28a** On the fresh pairs where names align >= 8 tensors, v4 with (g)-(i), the tie-break, (j)
  and (k), names ignored, gives the same related / unrelated verdict for >= 98% and the opposite
  verdict for none (judged if >= 20 pairs, >= 5 of them unrelated by names).
- **H28b** On the fresh declared pairs where names align fewer than 8 tensors and both models were
  read, `auto4` makes >= 80% comparable (>= 8 tensors).
- **H28c** No fresh sampled model, and no declared base of one, is unreadable because of its file
  format, except formats with no public specification; no weight files, gated, private or missing
  repositories are counted by reason and do not count.
- **H28d** On the fresh declared pairs of H28b whose two models share >= 8 tensors of >= 4096
  values by shape (either orientation), `auto4` makes >= 80% comparable (judged if >= 10 pairs).

**Outcome of H16b (recorded after judging, 2026-11-23; the 20,000-step students were trained on
friends' GPU accounts, `h16b-*`, and scored by `pedigree-e5e-h16b-*`, 22 students per family, none
failed).** **Not decidable**, as pre-registered when no family passes the floor (independents at
>= 50% of the teacher's task accuracy). Text encoder: the teacher scores 0.532 on the 2,425 masked
probe positions, and all 22 students, distilled and independent, score 0.0548 and agree with the
teacher on 0.0680, identical to four digits across 11 seeds and both kinds. Their training losses
level off after about a quarter of the 20,000 steps (independents near 7, distilled near 11), so the
students stopped at one output whatever the seed or the teacher, and behaviour has nothing of the
teacher to show. Vision transformer: the teacher scores 0.967 on CIFAR-10, and the 11 independents
score 0.275 to 0.377, under the floor of 0.484. No family is judged, so no rule is fixed and TPR
and FPR are not computed. H16 (supported) stands as the behaviour result for text decoders and
vision CNNs, and H18 stays not judged for its own reason, recorded above.

**Running E2y again (2026-11-23, before any E2y line is recorded).** Its second session
(`pedigree-e2y`, code 3340a12, readers v5, resuming the first session's raw rows) read 9 models,
204 pool pairs and 33 v4 pairs, and lost 1:
`cais/HarmBench-Llama-2-13b-cls` against `RichardErkhov/testmod_-_koen-llama2-13b-dpotrain_testver-gguf`,
"short read: 0 of 24 bytes". The cause is the repository, not the network. Twelve of its 24 GGUF files are
0 bytes and several others stop after a few hundred kilobytes, and the variant chosen (Q8_0) is one of the
empty ones. The server answers with a complete `200` and `Content-Length: 0`, so every rerun reads the
same nothing. E2x recorded this pair as not readable. Correction: `pedigree/remote.py`
now raises `FileTooShort` (a `ValueError`) when the response is complete, meaning its own
`Content-Length` arrived in full, and the file still ends before the bytes asked for. That is a
reason, not a loss. A body shorter than its own `Content-Length` is still a
dropped connection and is retried as before (`tests/test_remote_short.py`). This change only
reclassifies reads that were counted as lost, and a lost read holds back every ledger line of its
run, so no recorded verdict can change. E2y is run a third time, resuming the second session's raw
rows, so only the lost pair is read again.
*Addendum (2026-11-23, after the third session, before analysis):* the third session read the pair.
Once the empty Q8_0 file raises `FileTooShort`, the reader's existing rule for repositories with
several GGUF units (`units._most_tensors`: a unit whose header cannot be read is never preferred)
passes over it and reads `IQ3_M`, a complete upload (5,984,510,368 bytes). The pair compares 24
tensors at median 0.982. The session read 9 models, 204 pool pairs and 33 v4 pairs, lost none, and
recorded 246 ledger lines.

**Outcome of E2y (recorded after analysis, 2026-11-23; Kaggle `pedigree-e2y`, third session, read 9
models, 204 pool pairs and 33 v4 pairs, none lost).** E2y has no hypothesis of its own: it is E2 with
E2x and E2y applied, reported beside E2 as recorded (`verdicts.json` `readers_v5`). All four
models H21 found unreadable for reasons a reader could handle are now read: the three GGUF
repositories with quantisations in subfolders (`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`,
`AhmedPPL/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF`, `unsloth/Inkling-Small-GGUF`) and
`flair/chunk-english`. Of E2's 600 models, 5 stay not readable (was 9 with E2x), all by reason: no
weight files 3 (`Qdrant/bm25`, `qualcomm/DETR-ResNet101-DC5`,
`lmstudio-community/Qwen3-VL-8B-Instruct-MLX-8bit`), gated with no byte-identical public copy 1
(`Lightricks/LTX-2.5-22b-IC-LoRA-Deblur`), HTTP 403 1
(`arabsya87sy7a8/Qwythos-9B-Claude-Mythos-5-1M-GGUF`). Declared pairs not comparable fall from 24 to 21. Under the
published threshold, H2 reads 111/399 undeclared (E2x 111/394), H3 4/307 contradicted (3/301), and
H7 (a) 20/24, (b) 93/270 (92/266). Under the midpoint, H2 reads 108/387, H3 16/307 and H7 (b) 92/258.
No verdict changes from E2 as recorded. R2 puts 23 of the 27 declared pairs still not comparable into
a category; 4 are listed as gaps (`Comfy-Org/Qwen3-VL`, `tiny-random/qwen3.5-moe`,
`AhmedPPL/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF`, `KoreaPeter/ms-eff-gcvit-deepfake-b5-celeb-df-v2`).

**Running H28 (2026-11-23, before any H28 verdict).** The first session (`pedigree-h28`) read 149 of
the 150 sampled models in 2.7 hours, none lost (`results/kaggle/h28/h28_raw.jsonl`). It then stalled
on the 150th, `glogwa68/granite-4.0-h-1b-DISTILL-glm-4.7-think-GGUF`, until Kaggle ended the
session at its 12-hour limit. Reproduced locally, the model and its declared base read in seconds.
The stall is in one of its pool candidates, `taronaeo/Granite-4.0-H-1B-BE-GGUF`, a big-endian GGUF.
GGUF's specification allows big-endian files, and llama.cpp's `gguf_convert_endian.py` writes them.
Pedigree read every GGUF as little-endian. Version 3 big-endian then reads as 50,331,648, which passed
the version check, and the first string length read as about 1.4e18 bytes, so the reader fetched
block after block. Earlier runs met the same files and recorded them as not readable ("unpack
requires a buffer of 4 bytes": E2x's and E2y's pool pairs against `taronaeo/Granite-4.2-3B-BE-GGUF`
and `taronaeo/Granite-4.0-3B-Vision-BE-GGUF`).

*Correction to readers v6 (big-endian GGUF).* `pedigree/ggufremote.py` takes the byte order from
the version field, read in whichever order gives a plausible version, and reads the header in that
order. Tensor data is brought to little-endian by llama.cpp's own rule for each type:
- whole values swapped for F32, F16, BF16, F64 and the integer types;
- the fp16 fields of each block swapped for Q4_0, Q8_0, Q4_K and Q6_K;
- no change for MXFP4 and NVFP4.

A big-endian file of any other type is refused with that reason, because llama.cpp defines no byte
order for it. `tests/test_gguf_bigendian.py` writes the same weights little-endian and, with the
converter's own swaps, big-endian. Both read as the same values, and the test fails without the
block swaps. Under `PEDIGREE_READERS_V6=0` the earlier reading is kept, so H27 and E2y reproduce as
recorded. None of H28's 149 rows or its design pairs read a big-endian file, so they are not read
again. A second session resumes from the 149 and reads the last model with this correction.

**Outcome of H28 (recorded after judging, 2026-11-23; Kaggle `pedigree-h28`, two sessions, read all 150
sampled models, none lost; design data `pedigree-h28-design`, 32 pairs).**
- **H28a: supported.** Where names align at least 8 tensors, v4 with (g)-(i), the tie-break, (j) and (k),
  names ignored, gives the same verdict on 322 of 322 fresh pairs (23 of them unrelated by names). It
  gives the opposite verdict on none and aligns too few on none (v3: 317 of 322).
- **H28b: not supported.** 37 of 49 declared pairs where names align fewer than 8 tensors are comparable,
  under the 80% bar. One more declared pair was not read and counts under H28c.
- **H28c: supported.** No fresh model and no declared base is unreadable because of its file format. Five
  repositories are unread, all by reason: HTTP 401 3, gated with no byte-identical public copy 2.
- **H28d: supported.** Of H28b's pairs whose two models share at least 8 large tensor shapes, 37 of 44 are
  comparable. The 5 others in H28b share fewer than 8 such shapes, so no alignment can compare them.

H28d's 7 pairs not comparable, by measured cause:
- **A fork's own GGUF tensor types (2).** Types 100 and 101 are not in llama.cpp's published list, so they
  have no public specification
  (`kingjones777/DeepSeek-V4-Flash-180B-ROCmFP4-STRIX_LEAN-GGUF`, `kingjones777/Ling-3.0-flash-VL-MTP-ROCmFP4-GGUF`).
- **Draft or multi-token-prediction heads (3).** Each holds a few layers beside its declared full model:
  `a4lg/Qwen3-Next-80B-A3B-Instruct-MTP-ONLY-GGUF` 3 tensors compared,
  `kingjones777/Qwen3.8-Flash-Next-MTP-Heads-GGUF` 7, and the retrained EAGLE3 head
  `TiaoFeng/Spark-X2.6-4B-EAGLE3-Chinese-retrain` 1.
- **One tensor short (1).** `cstr/funasr-mlt-nano-GGUF` compares 7.
- **4-bit ONNX weights read as raw bytes (1).** `ibm-granite/granite-speech-5.0-2b-ONNX` stores its
  projections as onnxruntime `MatMulNBits` blocks: packed 4-bit `uint8` values with fp16 scales and zero
  points. The ONNX reader dequantises `DequantizeLinear` but not `MatMulNBits`, so the large weights stay
  unnamed `uint8` arrays, only the norms match by name, and nothing is compared. `MatMulNBits` is specified
  in onnxruntime's published contrib operators, so this is a reader the project still lacks. It is named
  here and not fixed after the verdict. It would not change one: H28b would read 38 of 49 and H28d
  38 of 44.

The design data (H27's opposite and not-comparable pairs, read with (ix) and (k)) is in the
`H28-design` ledger. In the 6 pairs where names now align at least 8 tensors (H27a's two opposite pairs
and four pairs where an imatrix had been taken for the model), v4 agrees with names on 6 of 6. The other
26 stay not comparable (0 of 26), for the causes measured under H27.

*Correction to the outcome of H28 (2026-11-23, after its verdicts; the verdicts above are unchanged).*
The cause given above for `ibm-granite/granite-speech-5.0-2b-ONNX` is wrong. Pedigree has read
`MatMulNBits` since reader (viii) (amendment 2026-11-21), and in this file it reads all 281 quantised
weights as values. What fails is the alignment. The file is written by onnxruntime-genai's model
builder, whose names separate the block index with underscores
(`model_layers_0_attn_q_proj_MatMul_weight_quant`). v3's and v4's block pattern accepts only `.` or `/`
around `layers.N`, so every weight falls in the group with no block index. There the repeated shapes
(q and o, k and v, gate and up in 40 layers) pair nothing, and the norms that names match are under
4,096 values. The pair is still one of H28d's 7 not comparable, still for a cause a rule can handle, and
H28b and H28d read as recorded. Measured with the rule below, the pair compares 24 tensors at median
0.997 under struct4 and auto4.

## Amendment 2026-11-23: rule (l) and H29, after H28 and before any of H29's data

H28's verdicts stand as recorded. Its pairs are design data for what follows, reported beside H29
and not counted.

**(l)** (`pedigree/structalign.py`, readers v7; `PEDIGREE_READERS_V7=0` reproduces H28 and everything
before it). In v3's and v4's block pattern, an underscore separates a block index as `.` and `/` do,
so `model_layers_0_attn_q_proj` is block 0 of stack `model_layers`. Names already read with `.` or
`/` give the same stack and block as before. `tests/test_rule_l.py` aligns onnxruntime-genai names
with Hugging Face names over 4 layers: all 28 projections pair with their own tensor with (l), and
none of them pair without it. The Kaggle jobs `h28` and `h28-design` now run with
`PEDIGREE_READERS_V7=0`.

**Fresh data.** A new frozen sample (`results/h29_sample.json`, seed 20261125, drawn after this
amendment is committed and before any of it is read): H25's rule, excluding E2's, H13's, H15's,
H19's, H23's, H25's, H26's, H27's and H28's samples. Each model is read and aligned exactly as in H28,
with (l). Design data (ledger `H29-design`): H28's 12 declared pairs where names aligned fewer than 8
tensors and auto4 left the pair not comparable, and any pair where v4 with names ignored gave a verdict
other than names' (none in H28 or H28-design).

- **H29a** On the fresh pairs where names align >= 8 tensors, v4 with (g)-(i), the tie-break, (j),
  (k) and (l), names ignored, gives the same related / unrelated verdict for >= 98% and the opposite
  verdict for none (judged if >= 20 pairs, >= 5 of them unrelated by names).
- **H29b** On the fresh declared pairs where names align fewer than 8 tensors and both models were
  read, `auto4` makes >= 80% comparable (>= 8 tensors).
- **H29c** No fresh sampled model, and no declared base of one, is unreadable because of its file
  format, except formats with no public specification; no weight files, gated, private or missing
  repositories are counted by reason and do not count.
- **H29d** On the fresh declared pairs of H29b whose two models share >= 8 tensors of >= 4096 values by
  shape (either orientation), `auto4` makes >= 80% comparable (judged if >= 10 pairs).

**H16c dev pilot (2026-11-23, design data; no H16c hypothesis is registered yet).** H16b's text-encoder
students all stopped at one output, with losses levelled near the unigram entropy, and E5's 2,000-step
students at 1e-4 levelled at the same loss. The usual cause in BERT-style (post-LN) models trained from
scratch at small batch is an unclipped update that pushes them into predicting token frequencies.
`experiments/h16c_pilot.py` trains independent half-depth students on the dev teacher only
(`bert-base-uncased`; the fresh RoBERTa family stays unread) for 5,000 steps, seed 90, under four
settings, and scores each on E5's probes as E5e does:
- H16b's own setting (rate 5e-4, 1,000 warm-up steps, batch 8, no clipping);
- the same with gradient-norm clipping at 1.0 (`PEDIGREE_CLIP`, default 0: no earlier run changes);
- rate 1e-4 with clipping;
- rate 2e-4 with clipping and batch 32.

The dev teacher scores 0.515 on the probes, so H16b's floor for independents would be 0.258. An H16c is
pre-registered only if a setting shows independents rising toward that floor. Otherwise the stall is
recorded with its measured cause, and H16b stays not decidable as recorded.
