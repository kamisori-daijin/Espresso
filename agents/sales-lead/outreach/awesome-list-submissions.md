# ESP-21: Awesome Lists, Newsletters & Product Hunt — Submission Package

## 1. awesome-swift (matteocrippa/awesome-swift)

**Repo**: https://github.com/matteocrippa/awesome-swift
**Action**: Open a PR adding Espresso under the AI/ML section
**PR Title**: `Add Espresso — direct ANE inference framework, 4.76x faster than CoreML`

### PR Description:
```
## Description

Adding [Espresso](https://github.com/christopherkarani/Espresso) to the Machine Learning section.

Espresso is a pure-Swift inference framework for Apple Silicon that achieves **4.76x faster** transformer inference than CoreML by accessing the Neural Engine directly via the private MIL text dialect. Zero dependencies, MIT licensed, Swift 6.2.

**Why it belongs in awesome-swift:**
- Pure Swift 6.2 (`~Copyable` move-only tensors, strict concurrency, typed throws)
- Zero external dependencies
- MIT licensed
- macOS 15+, all M-series chips (M1–M4) tested
- 926 tok/s on M3 Max — measurable, reproducible benchmark

**Entry:**
- [Espresso](https://github.com/christopherkarani/Espresso) - Direct Neural Engine inference framework for transformers. 4.76x faster than CoreML, pure Swift 6.2, zero dependencies.
```

### README diff (add under Machine Learning / AI section):
```diff
+- [Espresso](https://github.com/christopherkarani/Espresso) - Direct Neural Engine inference framework for transformers on Apple Silicon. 4.76x faster than CoreML, pure Swift 6.2, zero dependencies.
```

---

## 2. awesome-machine-learning (josephmisiti/awesome-machine-learning)

**Repo**: https://github.com/josephmisiti/awesome-machine-learning
**Action**: Open a PR adding Espresso under Swift section
**PR Title**: `Add Espresso to Swift section — direct ANE inference, 4.76x faster than CoreML`

### README diff (add under Swift subsection):
```diff
+#### Espresso
+
+[Espresso](https://github.com/christopherkarani/Espresso) - A pure-Swift inference framework for Apple Silicon that achieves direct Neural Engine access via the private MIL text dialect. Achieves 4.76x faster transformer inference than CoreML (926 tok/s on M3 Max), with zero external dependencies, MIT license, and full training support including backward passes and Adam optimizer.
```

### PR Description:
```
## Description

Adding Espresso to the Swift section under General-Purpose Machine Learning.

**Espresso** is an open-source (MIT), pure-Swift framework for Apple Silicon that bypasses CoreML and accesses the Neural Engine directly, enabling 4.76x faster transformer inference.

Key facts:
- Language: Swift 6.2 (strict concurrency, `~Copyable`, zero dependencies)
- Performance: 4.76x faster than CoreML / 926 tok/s on M3 Max
- Supports: M1, M2, M3, M4 chips (all ANE generations)
- Features: inference (fused decode with KV-cache) + training (full backward pass)
- License: MIT
- GitHub: https://github.com/christopherkarani/Espresso
```

---

## 3. awesome-apple-silicon (TacerJe/awesome-apple-silicon or similar)

**Repo**: https://github.com/TacerJe/awesome-apple-silicon (also check: https://github.com/mikeroyal/Apple-Silicon-Guide)
**Action**: Open a PR adding Espresso under ML Frameworks section
**PR Title**: `Add Espresso — Swift inference framework with 4.76x CoreML speedup`

### README diff:
```diff
+- **[Espresso](https://github.com/christopherkarani/Espresso)** — Pure-Swift direct ANE inference framework. Bypasses CoreML via private MIL text dialect. 4.76x faster, 926 tok/s on M3 Max. Zero dependencies. MIT.
```

---

## 4. iOS Dev Weekly — Newsletter Submission

**URL**: https://iosdevweekly.com/issues/submit (or email: dave@iosdevweekly.com)
**Subject**: `Espresso: Direct ANE inference in Swift — 4.76x faster than CoreML`

### Body:
```
Hi Dave,

I wanted to share a project that might be worth covering in iOS Dev Weekly:

**Espresso** — https://github.com/christopherkarani/Espresso

It's an open-source (MIT), pure-Swift framework for Apple Silicon that lets developers run transformer inference directly on the Neural Engine — bypassing CoreML entirely — achieving 4.76x faster performance (926 tok/s on M3 Max). Zero external dependencies, Swift 6.2, works across M1–M4.

What makes it technically interesting for iOS developers:
- Uses Swift 6.2 features: `~Copyable` move-only tensors, strict concurrency, typed throws
- Adds as a Swift Package in one line
- Works on macOS 15+ (iOS port in roadmap)
- Reproducible benchmarks included

The project demonstrates what's possible when you get below CoreML's abstraction layer on Apple Silicon. Given the current wave of on-device ML interest, I think developers would find the performance numbers and the pure-Swift approach genuinely surprising.

GitHub: https://github.com/christopherkarani/Espresso

Thanks for your consideration,
Chris Karani
```

---

## 5. Swift Weekly Brief — Newsletter Submission

**URL**: https://swiftweeklybrief.com/sponsorship/ or Swift Weekly Brief community form
**Method**: Submit via GitHub at https://github.com/SwiftWeekly/swiftweekly.github.io (open PR to add to issue links, or contact @jesse squires)
**Subject**: `Espresso: pure-Swift direct ANE inference, 4.76x faster than CoreML — worth a brief mention?`

### Body:
```
Hi Swift Weekly Brief team,

Wanted to flag a project that might be of interest to your readers:

**Espresso** — https://github.com/christopherkarani/Espresso

A pure-Swift 6.2 inference framework that achieves direct Apple Neural Engine access via the private MIL text dialect. It benchmarks 4.76x faster than CoreML for transformer inference (926 tok/s on M3 Max), with zero external dependencies and MIT license.

From a Swift language perspective, it's an interesting showcase of Swift 6.2:
- `~Copyable` move-only tensors for zero-copy I/O
- Strict concurrency via Swift 6.2 concurrency model
- Typed throws throughout
- `~Escapable` types for bounded lifetime IOSurface buffers

The performance story is compelling (4.76x over CoreML), but the Swift engineering approach might be equally interesting to your readers.

GitHub: https://github.com/christopherkarani/Espresso

Thanks,
Chris Karani
```

---

## 6. Product Hunt — Launch Strategy

**URL**: https://www.producthunt.com/posts/new
**Timing**: Tuesday or Wednesday, 12:01am PST (to maximize 24-hour voting window)
**Best window**: Week after WWDC announcements or Apple ML news cycle

### Tagline (60 chars max):
```
Neural Engine inference 4.76x faster than CoreML in Swift
```

### Description:
```
Espresso is a pure-Swift framework for Apple Silicon that runs transformer inference directly on the Neural Engine — bypassing CoreML's abstraction layer entirely.

**What it does:**
- 4.76x faster than CoreML (926 tok/s on M3 Max)
- Zero external dependencies — one SPM line to add
- Full training support: forward + backward passes, Adam optimizer
- Works across all M-series chips (M1–M4)

**How it works:**
Espresso compiles MIL (Model Intermediate Language) programs directly to ANE silicon through reverse-engineered private APIs. Instead of CoreML's per-token overhead, Espresso fuses 3 transformer layers into a single ANE dispatch — 6 layers in 2 evaluations instead of 6.

**Who it's for:**
- Swift/macOS developers building on-device AI
- ML engineers who want maximum ANE performance
- Researchers exploring Apple Silicon's actual inference ceiling

**Open source:** MIT licensed at https://github.com/christopherkarani/Espresso

```

### Gallery/Media needed:
1. Banner: benchmark comparison chart (Espresso vs CoreML vs llama.cpp)
2. Screenshot: TUI demo showing token generation
3. Screenshot: 5-line Swift code snippet for integration
4. Screenshot: Platform compatibility matrix

### First Comment (post immediately after launch to start conversation):
```
Hey PH! 👋 Founder here.

The quick backstory: CoreML is Apple's official ML framework, but it adds significant overhead between your model and the Neural Engine hardware. We reverse-engineered the private APIs to go direct, and the result is 4.76x the throughput.

The interesting engineering: instead of running 6 transformer layers in 6 separate ANE dispatches (CoreML's approach), Espresso fuses 3 layers into 1 dispatch using lane-packed attention kernels — so 6 layers = 2 evaluations.

Happy to answer any questions about the ANE internals, the MIL text dialect, or the Swift 6.2 patterns we used. This is a technically deep project and I love talking about it.

Repo: https://github.com/christopherkarani/Espresso
```

### Makers to add:
- @christopherkarani

### Topics:
- Developer Tools
- Machine Learning
- iOS
- Swift
- Open Source

---

## Execution Checklist

| Channel | Action | Status | Notes |
|---------|--------|--------|-------|
| awesome-swift | Open PR | Todo | matteocrippa/awesome-swift |
| awesome-machine-learning | Open PR | Todo | josephmisiti/awesome-machine-learning |
| awesome-apple-silicon | Open PR | Todo | TacerJe/awesome-apple-silicon + mikeroyal/Apple-Silicon-Guide |
| iOS Dev Weekly | Send email | Todo | dave@iosdevweekly.com |
| Swift Weekly Brief | GitHub PR or contact | Todo | SwiftWeekly/swiftweekly.github.io |
| Product Hunt | Schedule launch | Todo | Tue/Wed 12:01am PST, WWDC week ideal |

## Priority Order
1. **awesome-swift** (highest-traffic list, most relevant Swift devs)
2. **iOS Dev Weekly** (direct email, fastest turnaround)
3. **awesome-machine-learning** (broader ML audience)
4. **Swift Weekly Brief** (Swift-focused, curated)
5. **awesome-apple-silicon** (niche but targeted)
6. **Product Hunt** (requires media assets, time the launch)

