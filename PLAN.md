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
would change. A changed verdict is reported as a change, not as the verdict.

**E2p.** PyTorch files saved before torch 1.6 are a series of pickles followed by raw storages,
not a zip; the reader refused them ("legacy pickle files are not read"), which left 6 declared
bases unreadable. `pedigree/torchzip.py` now reads them with the same restricted unpickler
(PyTorch's tensor records only, any other global refused; `tests/test_torchzip.py` includes a
legacy file carrying code). E2p (`experiments/e2_rescore.py --match "legacy pickle" --name E2p`)
scores again every E2 comparison that ended in that error, and is applied beside E2 as E2r is.

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
