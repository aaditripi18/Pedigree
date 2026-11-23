"""Structural alignment of a GGUF file's tensors with a Hugging Face model's, for when
llama.cpp's name table does not know the architecture (PLAN.md, amendment 2026-11-17 (4), H17).

Both sides are grouped by block index (``blk.N.`` in GGUF; the first ``.layers.N.``,
``.layer.N.``, ``.h.N.``, ``.blocks.N.`` or ``.block.N.`` on the other side; tensors without an
index form one group). The two models must have the same number of blocks. Within a group a
tensor is paired only if its shape occurs exactly once in that group on both sides; ambiguous
shapes are left out, so a wrong pairing can only come from two different tensors that happen to
be the only ones of one shape in the same block on both sides.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict

_GGUF_BLOCK = re.compile(r"^blk\.(\d+)\.")
_HF_BLOCK = re.compile(r"(?:^|\.)(?:layers|layer|h|blocks|block)\.(\d+)\.")


def _block(name: str, gguf_side: bool) -> int | None:
    m = (_GGUF_BLOCK.match(name) if gguf_side else _HF_BLOCK.search(name))
    return int(m.group(1)) if m else None


def _groups(shapes: dict, gguf_side: bool) -> dict:
    """{block index or None: {shape: [names]}}"""
    g: dict = defaultdict(lambda: defaultdict(list))
    for name, (_kind, shape) in shapes.items():
        g[_block(name, gguf_side)][tuple(shape)].append(name)
    return g


def struct_names(gguf_shapes: dict, hf_shapes: dict) -> dict[str, str]:
    """{gguf tensor name: hf tensor name}, paired by block index and unique shape.
    Shapes are dicts {name: (kind, shape)} with the GGUF shape already in Hugging Face order."""
    g, h = _groups(gguf_shapes, True), _groups(hf_shapes, False)
    nb_g = len([k for k in g if k is not None])
    nb_h = len([k for k in h if k is not None])
    if nb_g != nb_h:
        return {}
    out = {}
    for key, gs in g.items():
        hs = h.get(key)
        if not hs:
            continue
        for shape, names in gs.items():
            other = hs.get(shape)
            if len(names) == 1 and other and len(other) == 1:
                out[names[0]] = other[0]
    return out


# --- v2 (PLAN.md, amendment 2026-11-18, H20): stacks, so unequal block counts still align ----------

_GGUF_STACK = re.compile(r"^(?:(.+)\.)?blk\.(\d+)\.")
_HF_STACK = re.compile(r"(?:^|\.)(layers|layer|h|blocks|block)\.(\d+)\.")


def _stack(name: str, gguf_side: bool) -> tuple[str, int] | None:
    if gguf_side:
        m = _GGUF_STACK.match(name)
        return ((m.group(1) + ".blk") if m.group(1) else "blk", int(m.group(2))) if m else None
    m = _HF_STACK.search(name)
    return (name[:m.end(1)], int(m.group(2))) if m else None


def _stacks(shapes: dict, gguf_side: bool):
    """({stack: {block: {shape: [names]}}}, {shape: [names]} for tensors without a block index)"""
    st: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    free: dict = defaultdict(list)
    for name, (_kind, shape) in shapes.items():
        s = _stack(name, gguf_side)
        if s is None:
            free[tuple(shape)].append(name)
        else:
            st[s[0]][s[1]][tuple(shape)].append(name)
    return st, free


def _unique_pairs(gs: dict, hs: dict, out: dict) -> None:
    for shape, names in gs.items():
        other = hs.get(shape)
        if len(names) == 1 and other and len(other) == 1:
            out[names[0]] = other[0]


def struct_names_v2(gguf_shapes: dict, hf_shapes: dict) -> dict[str, str]:
    """{gguf tensor name: hf tensor name}. GGUF stacks, largest first, each paired with the unpaired
    stack of the other side with the same block count, else the nearest (ties: more tensors); block
    i with block i below the smaller count; unique shapes within each block, and among tensors with
    no block index."""
    gst, gfree = _stacks(gguf_shapes, True)
    hst, hfree = _stacks(hf_shapes, False)

    def size(st, k):
        return sum(len(v) for blk in st[k].values() for v in blk.values())
    out: dict = {}
    free_h = set(hst)
    for g in sorted(gst, key=lambda k: -size(gst, k)):
        if not free_h:
            break
        ng = len(gst[g])
        h = min(free_h, key=lambda k: (abs(len(hst[k]) - ng), -size(hst, k), *_tie(g, k)))
        free_h.discard(h)
        for i in range(min(len(gst[g]), len(hst[h]))):     # block i with block i
            if i in gst[g] and i in hst[h]:
                _unique_pairs(gst[g][i], hst[h][i], out)
    _unique_pairs(gfree, hfree, out)
    return out


# --- v3 (PLAN.md, amendment 2026-11-20, H23): any two formats, roles, unequal depth --------------

_ANY_STACK = re.compile(r"(?:^|[./])(blk|layers|layer|h|blocks|block|layer_)[._]?(\d+)(?=[./])")
# (l) (amendment 2026-11-23, readers v7): an underscore also separates a block index, as in the names
# onnxruntime-genai's model builder gives (``model_layers_0_attn_q_proj_MatMul_weight_quant``). Without
# it every tensor of such a file fell in the group with no block index, where repeated shapes pair
# nothing. PEDIGREE_READERS_V7=0 reproduces H28 and everything before it.
READERS_V7 = os.environ.get("PEDIGREE_READERS_V7", "1") != "0"
_ANY_STACK_L = re.compile(r"(?:^|[./_])(blk|layers|layer|h|blocks|block|layer_)[._]?(\d+)(?=[./_])")


def _stack_re():
    return _ANY_STACK_L if READERS_V7 else _ANY_STACK


_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "model.", "transformer.")
ROLES = {"encoder": {"encoder", "enc"}, "decoder": {"decoder", "dec"},
         "vision": {"vision", "visual", "v", "vit", "image"}, "text": {"text", "t", "language_model", "llm"},
         "audio": {"audio", "a"}}


def _strip(name: str) -> str:
    for p in _PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def direct_names(a_shapes: dict, b_shapes: dict) -> dict[str, str]:
    """(a) equal names after removing one leading model./diffusion_model./transformer. prefix,
    with equal shapes."""
    b = {}
    for n, (_k, s) in b_shapes.items():
        b.setdefault(_strip(n), []).append((n, tuple(s)))
    out = {}
    for n, (_k, s) in a_shapes.items():
        hit = [m for m, t in b.get(_strip(n), []) if t == tuple(s)]
        if len(hit) == 1:
            out[n] = hit[0]
    return out


def _stack3(name: str):
    m = _stack_re().search(name)
    return (name[:m.end(1)], int(m.group(2))) if m else None


def _stacks3(shapes: dict):
    st: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    free: dict = defaultdict(list)
    for name, (_kind, shape) in shapes.items():
        s = _stack3(name)
        if s is None:
            free[tuple(shape)].append(name)
        else:
            st[s[0]][s[1]][tuple(shape)].append(name)
    return st, free


def _role(prefix: str) -> str | None:
    toks = set(re.split(r"[./_]", prefix.lower()))
    found = [r for r, words in ROLES.items() if toks & words]
    return found[0] if len(found) == 1 else None


def _transposed_pairs(gs: dict, hs: dict, out: dict, tr: set) -> None:
    """(d) a 2-D tensor whose shape occurs once on each side only when reversed: its transpose."""
    for shape, names in gs.items():
        if len(shape) != 2 or shape[0] == shape[1] or len(names) != 1 or names[0] in out:
            continue
        if shape in hs:
            continue
        other = hs.get(shape[::-1])
        if other and len(other) == 1 and other[0] not in out.values():
            out[names[0]] = other[0]
            tr.add(names[0])


def struct_names_v3(a_shapes: dict, b_shapes: dict) -> tuple[dict[str, str], set]:
    """({a tensor: b tensor}, {a tensors paired as transposes}). Stacks paired by a shared role
    word first, then as in v2 (same block count, else nearest; ties: more tensors); block i with
    block round(i * (m - 1) / (n - 1)) when the counts n and m differ; unique shapes within each
    block pair and among tensors with no block index; else unique reversed 2-D shapes."""
    ast_, afree = _stacks3(a_shapes)
    bst, bfree = _stacks3(b_shapes)

    def size(st, k):
        return sum(len(v) for blk in st[k].values() for v in blk.values())
    out, tr = {}, set()
    free_b = set(bst)
    order = sorted(ast_, key=lambda k: (_role(k) is None, -size(ast_, k)))
    for g in order:
        if not free_b:
            break
        ng = len(ast_[g])
        same_role = [k for k in free_b if _role(g) and _role(k) == _role(g)]
        pool_ = same_role or ([k for k in free_b if _role(k) is None or _role(g) is None] or list(free_b))
        h = min(pool_, key=lambda k: (abs(len(bst[k]) - ng), -size(bst, k), *_tie(g, k)))
        free_b.discard(h)
        ga, hb = sorted(ast_[g]), sorted(bst[h])
        n, m = len(ga), len(hb)
        for i, bi in enumerate(ga):
            j = i if n == m else (round(i * (m - 1) / (n - 1)) if n > 1 else 0)
            if j < m:
                _unique_pairs(ast_[g][bi], bst[h][hb[j]], out)
                _transposed_pairs(ast_[g][bi], bst[h][hb[j]], out, tr)
    _unique_pairs(afree, bfree, out)
    _transposed_pairs(afree, bfree, out, tr)
    return out, tr


def names_v3(a_shapes: dict, b_shapes: dict) -> tuple[dict[str, str], set]:
    """H23's v3: direct names (a), else role-paired stacks with unequal depth (b, c, d), whichever
    pairs more tensors."""
    d = direct_names(a_shapes, b_shapes)
    s, tr = struct_names_v3(a_shapes, b_shapes)
    return (s, tr) if len(s) > len(d) else (d, set())


# --- v4 (PLAN.md, amendment 2026-11-20 (2), H25): size-1 dimensions, repeated shapes, depth maps -----

def _sq(shape) -> tuple:
    """(e) a shape without its size-1 dimensions (the element order is unchanged)."""
    return tuple(d for d in shape if d != 1)


def _stacks4(shapes: dict):
    st: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    free: dict = defaultdict(list)
    for name, (_kind, shape) in shapes.items():
        s = _stack3(name)
        if s is None:
            free[_sq(shape)].append(name)
        else:
            st[s[0]][s[1]][_sq(shape)].append(name)
    return st, free


def _leaf(name: str) -> str:
    """The part of a name after its block index (or the whole name, prefix removed), lower case,
    letters and digits only."""
    m = _stack_re().search(name)
    rest = name[m.end():] if m else _strip(name)
    return re.sub(r"[^a-z0-9]", "", rest.lower())


def _lcs(a: str, b: str) -> int:
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def _tie(a: str, b: str) -> tuple:
    """The last tie-break between candidate partner stacks (amendment 2026-11-22): the stack name
    sharing the longest subsequence with the other side's, then name order. Without it a tie was
    settled by a set's iteration order, which Python varies between runs (H26a's opposite verdict)."""
    return (-_lcs(a.lower(), b.lower()), b)


def _pairs4(gs: dict, hs: dict, out: dict) -> None:
    """Unique shapes as v1; (f) a shape that repeats: a and b are paired when each is the other's
    single best match by leaf-name LCS length, strictly above its second best."""
    used = set(out.values())
    for shape, names in gs.items():
        other = hs.get(shape)
        if not other:
            continue
        A = [x for x in names if x not in out]
        B = [y for y in other if y not in used]
        if not A or not B:
            continue
        if len(names) == 1 and len(other) == 1:
            out[A[0]] = B[0]
            used.add(B[0])
            continue
        la, lb = {x: _leaf(x) for x in A}, {y: _leaf(y) for y in B}
        S = {(x, y): _lcs(la[x], lb[y]) for x in A for y in B}
        for x in A:
            sx = sorted((S[x, y] for y in B), reverse=True)
            if len(sx) > 1 and sx[0] == sx[1]:
                continue
            y = max(B, key=lambda y: S[x, y])
            sy = sorted((S[x2, y] for x2 in A), reverse=True)
            if (len(sy) > 1 and sy[0] == sy[1]) or sy[0] != S[x, y] or y in used:
                continue
            out[x] = y
            used.add(y)


def direct_names4(a_shapes: dict, b_shapes: dict) -> dict[str, str]:
    """(a) with (e): equal names after the prefix, equal shapes without size-1 dimensions."""
    b: dict = {}
    for n, (_k, s) in b_shapes.items():
        b.setdefault(_strip(n), []).append((n, _sq(s)))
    out = {}
    for n, (_k, s) in a_shapes.items():
        hit = [m for m, t in b.get(_strip(n), []) if t == _sq(s)]
        if len(hit) == 1:
            out[n] = hit[0]
    return out


DEPTH_MAPS = ("first", "last", "proportional")


def _block_map(kind: str, n: int, m: int, i: int) -> int | None:
    if n == m or kind == "first":
        j = i
    elif kind == "last":
        j = i + m - n
    else:
        j = round(i * (m - 1) / (n - 1)) if n > 1 else 0
    return j if 0 <= j < m else None


def _shape_counts(stack: dict) -> dict:
    out: dict = defaultdict(int)
    for blk in stack.values():
        for shape, names in blk.items():
            out[shape] += len(names)
    return out


def _overlap(a: dict, b: dict) -> int:
    """How many tensors two stacks could pair by shape (size-1 dimensions aside): the multiset
    intersection of their shapes."""
    return sum(min(n, b.get(s, 0)) for s, n in a.items())


def struct_names_v4(a_shapes: dict, b_shapes: dict) -> list[tuple[str, dict, set]]:
    """[(depth map, {a tensor: b tensor}, {a tensors paired as transposes})]: one entry if every
    paired stack has equal depth, else one per map in DEPTH_MAPS (c4). Stacks are paired as in v3."""
    ast_, afree = _stacks4(a_shapes)
    bst, bfree = _stacks4(b_shapes)

    def size(st, k):
        return sum(len(v) for blk in st[k].values() for v in blk.values())
    stack_pairs, free_b = [], set(bst)
    for g in sorted(ast_, key=lambda k: (_role(k) is None, -size(ast_, k))):
        if not free_b:
            break
        ng = len(ast_[g])
        same_role = [k for k in free_b if _role(g) and _role(k) == _role(g)]
        pool_ = same_role or ([k for k in free_b if _role(k) is None or _role(g) is None] or list(free_b))
        if V5:   # (i) the partner whose tensor shapes overlap most, then the old rule (amendment 2026-11-21)
            sg = _shape_counts(ast_[g])
            h = min(pool_, key=lambda k: (-_overlap(sg, _shape_counts(bst[k])), abs(len(bst[k]) - ng), -size(bst, k),
                                          *_tie(g, k)))
        else:
            h = min(pool_, key=lambda k: (abs(len(bst[k]) - ng), -size(bst, k), *_tie(g, k)))
        free_b.discard(h)
        stack_pairs.append((g, h))
    unequal = any(len(ast_[g]) != len(bst[h]) for g, h in stack_pairs)
    out = []
    for kind in (DEPTH_MAPS if unequal else ("equal",)):
        mp, tr = {}, set()
        for g, h in stack_pairs:
            ga, hb = sorted(ast_[g]), sorted(bst[h])
            for i, bi in enumerate(ga):
                j = _block_map(kind, len(ga), len(hb), i)
                if j is not None:
                    _pairs4(ast_[g][bi], bst[h][hb[j]], mp)
                    _transposed_pairs(ast_[g][bi], bst[h][hb[j]], mp, tr)
        _pairs4(afree, bfree, mp)
        _transposed_pairs(afree, bfree, mp, tr)
        out.append((kind, mp, tr))
    return out


# --- amendment 2026-11-21: (g) and (h), with readers v5 (PEDIGREE_READERS_V5=0 reproduces H25) -----

V5 = os.environ.get("PEDIGREE_READERS_V5", "1") != "0"
_ANON = re.compile(r"^(?:t|tensor[_.]?)?(\d+)$", re.I)
_PEFT = re.compile(r"\.base_layer(?=\.)")


def positional_names(a_shapes: dict, b_shapes: dict) -> list[tuple[str, dict]]:
    """(g) One side names its tensors by number only (a converter that dropped the names): its
    tensors, in number order, are paired by position with the other side's, in header order and in
    name order, each order kept only when the two shape sequences are equal (size-1 dimensions
    aside) throughout. Where both orders qualify, each is a candidate, scored like (c4)'s maps."""
    out = []
    for x, y, flip in ((a_shapes, b_shapes, False), (b_shapes, a_shapes, True)):
        if len(x) < 2 or len(x) != len(y) or not all(_ANON.match(n) for n in x):
            continue
        xs = sorted(x, key=lambda n: int(_ANON.match(n).group(1)))
        for label, order in (("positional-header", list(y)), ("positional-name", sorted(y))):
            if all(_sq(x[p][1]) == _sq(y[q][1]) for p, q in zip(xs, order)):
                m = dict(zip(xs, order))
                m = {v: k for k, v in m.items()} if flip else m
                if all(m != o for _, o in out):
                    out.append((label, m))
        if out:
            return out
    return out


def prefix_names(a_shapes: dict, b_shapes: dict, big=lambda names: len(names)) -> dict[str, str]:
    """(h) One model holds the other under a leading path (`llm.base_model.model.` + the base's own
    names): the path most often left when a name is cut down to one on the other side is removed
    from every name, PEFT's `.base_layer` (an unmerged adapter's base weight) is read as the weight
    itself, and names then equal, with shapes equal (size-1 dimensions aside), are paired."""
    from collections import Counter
    best: dict = {}
    for x, y, flip in ((a_shapes, b_shapes, False), (b_shapes, a_shapes, True)):
        yn: dict = {}
        for n in y:
            yn.setdefault(_PEFT.sub("", n), []).append(n)
        cut = Counter()
        for n in x:
            parts = _PEFT.sub("", n).split(".")
            for k in range(1, len(parts) - 1):
                if len(yn.get(".".join(parts[k:]), [])) == 1:
                    cut[".".join(parts[:k])] += 1
                    break
        if not cut:
            continue
        pre = cut.most_common(1)[0][0] + "."
        m = {}
        for n in x:
            nn = _PEFT.sub("", n)
            hit = yn.get(nn[len(pre):], []) if nn.startswith(pre) else []
            if len(hit) == 1 and _sq(x[n][1]) == _sq(y[hit[0]][1]):
                m[n] = hit[0]
        if flip:
            m = {v: k for k, v in m.items()}
        if big(m) > big(best):
            best = m
    return best


def names_v4(a_shapes: dict, b_shapes: dict, big=lambda names: len(names)) -> list[tuple[str, dict, set]]:
    """H25's v4: direct names (a, e), else the structural candidates (one per depth map), whichever
    pairs more large tensors (`big` counts them). With v5, (g) positional and (h) prefix pairings
    compete with direct names on the same count."""
    best = [("direct", direct_names4(a_shapes, b_shapes))]
    if V5:
        for group in (positional_names(a_shapes, b_shapes), [("prefix", prefix_names(a_shapes, b_shapes, big))]):
            if group and max(big(m) for _, m in group) > max(big(m) for _, m in best):
                best = group
    cands = struct_names_v4(a_shapes, b_shapes)
    if cands and max(big(m) for _, m, _ in cands) > max(big(m) for _, m in best):
        return cands
    return [(label, m, set()) for label, m in best]
