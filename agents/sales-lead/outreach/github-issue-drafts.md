# GitHub Issue / Discussion Drafts — Ready to Post

These are copy-paste ready messages for each GitHub target.
Post these manually or via `gh issue create` / `gh discussion create`.

---

## 1. Stephen Panaro — smpanaro/more-ane-transformers

**Repo**: https://github.com/smpanaro/more-ane-transformers
**Action**: Open a new Discussion (GitHub Discussions) or Issue titled:

> `Espresso: 4.76x over CoreML — benchmark comparison?`

**Body**:
```
Hi @smpanaro,

I've been following your work on more-ane-transformers closely — especially your GPT-2-xl results on the ANE. Really excellent engineering.

I wanted to share a related project: **Espresso** (https://github.com/christopherkarani/Espresso), a pure-Swift inference framework that achieves 4.76x faster than CoreML via direct ANE access using the private MIL text dialect. We've been focusing on the fused-kernel path for decode with KV-cache, reaching ~1.93ms/tok (519 tok/s) on M3 Max.

A few things I'd be curious to get your take on:
1. How does our fused decode approach compare to your ANE pipeline? (We use lane-packed attention kernels and triplet-layer fusion)
2. Would you be open to running the Espresso benchmarks on your hardware for comparison?
3. Any architectural decisions you've found particularly important that we might be missing?

Happy to share detailed technical notes on our MIL approach — particularly the lane-packed attention kernel design and fused RWKV-style recurrent decode that got us to 3.41x over CoreML baseline.

Thanks for the great work you've been doing in this space.

— Chris (@christopherkarani)
```

**Post as**: Discussion (category: Show and tell, or General)

---

## 2. Hollance — hollance/neural-engine

**Repo**: https://github.com/hollance/neural-engine
**Action**: Open a new Issue titled:

> `Espresso: production ANE inference framework + new MIL gotchas to contribute`

**Body**:
```
Hi @hollance,

Your "Everything we actually know about the Apple Neural Engine" repo has been an invaluable reference — thank you for documenting all of that. It's been essential to our research.

I'm building **Espresso** (https://github.com/christopherkarani/Espresso), a pure-Swift inference framework for Apple Silicon that uses the private ANE APIs you've documented to achieve 4.76x faster inference than CoreML (519 tok/s on M3 Max).

We've validated much of what you've documented and discovered a few additional gotchas we'd love to contribute back:

- `softmax` on non-power-of-2 dimensions → `InvalidMILProgram` at compile time
- `slice_by_index` on function inputs combined with RMSNorm+convs → `InvalidMILProgram` (workaround: prepare data in the right layout before passing to the function)
- Lane-packed attention kernels (spatial=32) necessary for stable ANE eval across M1-M4
- `reduce_mean` does NOT exist in raw MIL text format — use `reduce_sum` + `mul` by 1/N
- ANE eval unstable on some hosts even for single-input identity kernels

Would you be open to:
1. Adding a reference to Espresso in the README as a production inference usage example?
2. A PR contributing these findings to your docs?

Happy to submit the PR regardless of the README link. This community benefits from shared knowledge, and your docs deserve to stay current.

— Chris
https://github.com/christopherkarani/Espresso
```

**Post as**: Issue (label: enhancement or documentation)

---

## 3. Maderix — maderix/ANE

**Repo**: https://github.com/maderix/ANE
**Action**: Open a new Discussion or Issue titled:

> `Espresso + your ANE training work = complete ANE ecosystem?`

**Body**:
```
Hi @maderix,

I've been following your Substack series "Inside the M4 Apple Neural Engine" and your work on ANE training (maderix/ANE). The efficiency numbers and your findings on the convolution vs. matmul throughput differences are impressive.

I'm working on the inference side of the same problem space with **Espresso** (https://github.com/christopherkarani/Espresso), a pure-Swift framework that achieves 519 tok/s on M3 Max for transformer inference using the same private MIL APIs (3.41x over CoreML).

Our experiences complement each other naturally:
- You've mapped ANE training primitives and the full software stack to the IOKit layer
- We've mapped ANE inference kernel fusion patterns (fused RWKV-style decode, lane-packed attention, triplet-layer fusion)

There's a natural joint story here: a combined piece on "The complete ANE developer toolkit — training and inference." The Hacker News thread on your Part 1 shows the community is hungry for this kind of deep work.

Would you be interested in:
1. A joint blog post comparing our benchmark methodologies and results?
2. Cross-referencing findings (your 19 TFLOPS FP16, our 519 tok/s decode path)?
3. Exploring whether Espresso could serve as an inference runtime for models trained via your framework?

This is a genuinely novel space and combining our research could make a strong contribution to the community.

— Chris
https://github.com/christopherkarani/Espresso
```

**Post as**: Discussion (category: General)

---

## 4. MLX Team — ml-explore/mlx-swift

**Repo**: https://github.com/ml-explore/mlx-swift
**Action**: Open a new Discussion titled:

> `Espresso: direct ANE inference at 4.76x CoreML — potential complementary approach?`

**Body**:
```
Hi MLX team,

I wanted to share **Espresso** (https://github.com/christopherkarani/Espresso) and get your technical perspective.

Espresso is an open-source (MIT), pure-Swift framework for Apple Silicon that accesses the ANE directly via the private MIL text dialect — bypassing CoreML's abstraction layer. On M3 Max we achieve:
- 4.76x faster inference than standard CoreML
- 519 tok/s for transformer inference with fused KV-cache decode
- 1.93 ms/token end-to-end

I see Espresso as potentially complementary to MLX rather than competitive — MLX provides an excellent general-purpose ML framework, while Espresso demonstrates the maximum possible performance ceiling for ANE inference on specific transformer architectures.

A few questions for the team:
1. Are there aspects of the MIL approach that could inform MLX's ANE dispatch path?
2. Is there interest in a community benchmark that compares MLX-swift vs. Espresso vs. CoreML for transformer inference?
3. Any technical feedback on our kernel fusion approach?

Happy to discuss further in this thread or via a GitHub issue.

— Chris
https://github.com/christopherkarani/Espresso
```

**Post as**: Discussion (category: General)

---

## 5. ANEMLL — Anemll/anemll-bench (or Anemll/Anemll)

**Repo**: https://github.com/Anemll/Anemll
**Action**: Open a new Issue titled:

> `Add Espresso to ANEMLL benchmark suite?`

**Body**:
```
Hi ANEMLL team,

I'd like to propose adding **Espresso** (https://github.com/christopherkarani/Espresso) to your ANE benchmark suite.

Espresso is a pure-Swift inference framework that achieves direct ANE access via the private MIL text dialect, reaching:
- **4.76x faster than CoreML** on M3 Max
- **519 tok/s** (1.93 ms/token) for transformer inference with fused KV-cache decode
- Zero external dependencies, MIT licensed

I think this would make a strong addition to your benchmark suite — it represents the current performance ceiling for direct ANE inference, and including it would give your users a clear reference point for what's achievable without CoreML's abstraction layer.

I'm happy to:
- Contribute the benchmark scripts directly via PR
- Provide benchmark results across different Apple Silicon chips (M1/M2/M3/M4)
- Help verify correctness of the benchmark methodology

Would you be open to including Espresso in the suite?

— Chris
https://github.com/christopherkarani/Espresso
```

**Post as**: Issue

---

## Posting Instructions

```bash
# GitHub CLI — post as issue (example for target 2)
gh issue create \
  --repo hollance/neural-engine \
  --title "Espresso: production ANE inference framework + new MIL gotchas to contribute" \
  --body "$(cat agents/sales-lead/outreach/github-issue-drafts.md | <extract hollance body>)"

# Or use GitHub web UI directly — copy the body text above
```

## Timing
- Best time to post: Weekday mornings 9am-noon PST (Tuesday-Thursday preferred)
- GitHub issues get best engagement when posted Tuesday or Wednesday
