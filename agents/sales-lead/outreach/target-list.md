# Espresso Developer Outreach — Target List

**Goal**: Get 10 influential developers to try Espresso and share their experience
**Deadline**: Ongoing — start immediately
**Status**: Prepared, pending execution

---

## Tier 1: Direct ANE / Apple Silicon ML Community (Highest Priority)

### 1. Stephen Panaro (@smpanaro)
- **Why**: Built `more-ane-transformers` — GPT-2-xl on ANE. Direct overlap with Espresso's approach.
- **Channel**: GitHub issue/discussion on `smpanaro/more-ane-transformers`
- **Angle**: "Espresso achieves 4.76x over CoreML using similar ANE techniques. I'd love your benchmark comparison and thoughts on the approach."
- **Ask**: Run Espresso on their benchmarks, share results, potential co-authorship on technical post
- **Priority**: CRITICAL — direct peer with credibility in the space

### 2. Hollance (matthijs@...achine.dev)
- **Why**: Wrote the definitive guide "Everything we know about the Apple Neural Engine" (2.2K+ stars). Espresso is a practical implementation of concepts they documented.
- **Channel**: GitHub (@hollance) on `hollance/neural-engine`
- **Angle**: "Espresso is a production-ready inference framework built on the ANE fundamentals you documented. Would love your feedback and to link from your README."
- **Ask**: README cross-link, technical review, potential blog post
- **Priority**: HIGH — authoritative voice, low friction ask

### 3. Maderix (@maderix)
- **Why**: Reverse-engineered ANE for training (Substack: "Inside the M4 Apple Neural Engine"). Complementary: they do training, we do inference.
- **Channel**: GitHub `maderix/ANE` issues or Substack contact
- **Angle**: "Espresso is the inference-side complement to your training work. Interested in a joint technical post comparing approaches and benchmarks?"
- **Ask**: Joint blog post, cross-cite benchmarks, shared audience
- **Priority**: HIGH — direct community credibility, good story for both projects

---

## Tier 2: Apple ML Ecosystem

### 4. MLX Core Contributors (ml-explore/mlx)
- **Why**: Apple's official open-source ML framework. Espresso demonstrates what's possible at the ANE layer beneath MLX.
- **Channel**: GitHub `ml-explore/mlx` discussion or `ml-explore/mlx-swift`
- **Angle**: "Espresso reaches 4.76x over CoreML via direct ANE access. Would love feedback on our MIL approach and whether there's a path to upstream these optimizations."
- **Ask**: Technical feedback, potential feature collaboration
- **Priority**: MEDIUM — influence is high, response rate is uncertain (Apple internal)

### 5. Hugging Face Swift/CoreML Team
- **Why**: Owns the CoreML model conversion pipeline for the HF Hub. Could promote Espresso as a deployment target.
- **Channel**: GitHub `apple/coremltools` issues, HF Discord (#apple-silicon)
- **Angle**: "Espresso provides 4.76x faster inference than standard CoreML deployment. Interested in exploring an Espresso export path for transformers on the Hub?"
- **Ask**: Discussion about HF Hub integration, potential model collection
- **Priority**: HIGH — multiplier effect through HF ecosystem

---

## Tier 3: Swift Developer Influencers

### 6. Paul Hudson (@twostraws, hackingwithswift.com)
- **Why**: ~170K YouTube subscribers. Most-read Swift educator. Would create legitimacy with broader iOS developer audience.
- **Channel**: Twitter/X @twostraws, email via hackingwithswift.com
- **Angle**: "Espresso lets iOS/macOS developers run ML inference 4.76x faster than CoreML. Would this be interesting for a Hacking with Swift tutorial or Swift News episode?"
- **Ask**: Tutorial feature, Swift News mention, potential sponsorship discussion
- **Priority**: HIGH — reach is massive, but requires relevance to his teaching focus

### 7. Sean Allen (@seanallen_dev)
- **Why**: ~170K YouTube subscribers, iOS education focus, Swift News show
- **Channel**: Twitter/X @seanallen_dev
- **Angle**: Similar to Paul Hudson — performance story, AI/ML on Apple hardware is trending
- **Ask**: Swift News segment, tutorial collaboration
- **Priority**: MEDIUM — good reach, somewhat different audience (learners vs. advanced devs)

---

## Tier 4: Academic & Research

### 8. Apple ml-ane-transformers maintainer
- **Why**: Official Apple reference implementation. Getting cited by Apple would be significant.
- **Channel**: GitHub `apple/ml-ane-transformers` discussions
- **Angle**: "Espresso builds on the optimization patterns in your reference implementation and achieves [X]x over standard ANE usage. Would appreciate your review."
- **Ask**: Citation/reference in Apple docs, technical discussion
- **Priority**: MEDIUM-LOW — high value, low success probability

### 9. ANEMLL Team (anemll-bench)
- **Why**: ANE-specific benchmarking project. Adding Espresso to their benchmark would expose it to their audience.
- **Channel**: GitHub `Anemll/anemll-bench` issues
- **Angle**: "Would you consider adding Espresso to your ANE benchmark suite? Happy to contribute the benchmark scripts."
- **Ask**: Benchmark inclusion, mutual promotion
- **Priority**: MEDIUM — targeted audience, good mutual benefit

### 10. Independent iOS Performance Bloggers
- **Why**: Technical deep-dives on iOS performance reach the exact audience that would use Espresso
- **Outlets**: Substack, Medium (Towards Data Science), personal blogs
- **Approach**: Pitch a guest post or request a review/feature
- **Priority**: MEDIUM — scale through long-tail distribution

---

## Contact Details

| # | Target | Handle / Contact | Channel |
|---|--------|-----------------|---------|
| 1 | Stephen Panaro | @smpanaro (GitHub), @flat (Twitter) | github.com/smpanaro/more-ane-transformers |
| 2 | Matthijs Hollemans | @hollance (GitHub), @mhollemans | github.com/hollance/neural-engine |
| 3 | Maderix | @maderix (GitHub/Substack) | maderix.substack.com, github.com/maderix/ANE |
| 4 | MLX Team | ml-explore org | github.com/ml-explore/mlx-swift |
| 5 | HF Swift Team | Hugging Face | github.com/apple/coremltools |
| 6 | Paul Hudson | @twostraws | paul@hackingwithswift.com |
| 7 | Sean Allen | @seanallen_dev | Twitter DM (no public email) |
| 8 | Apple ANE team | - | github.com/apple/ml-ane-transformers |
| 9 | ANEMLL | Anemll org | github.com/Anemll/Anemll |
| 10 | Performance bloggers | Various | Substack/Medium pitch |

## Tracking

| # | Target | Status | Date Contacted | Response | Notes |
|---|--------|--------|---------------|----------|-------|
| 1 | Stephen Panaro | Sent | 2026-03-17 | Awaiting | github.com/smpanaro/more-ane-transformers/discussions/4 |
| 2 | Hollance | Sent | 2026-03-17 | Awaiting | github.com/hollance/neural-engine/issues/44 |
| 3 | Maderix | Sent | 2026-03-17 | Awaiting | github.com/maderix/ANE/issues/49 |
| 4 | MLX Contributors | Sent | 2026-03-17 | Awaiting | github.com/ml-explore/mlx-swift/discussions/372 |
| 5 | HF Swift Team | Draft ready | - | - | See messages.md |
| 6 | Paul Hudson | Email drafted | 2026-03-16 | - | Gmail draft: r-5734871108926968060 |
| 7 | Sean Allen | Draft ready | - | - | Twitter DM — use messages.md template |
| 8 | Apple ANE team | Backlog | - | - | Low priority — post after tier 1-2 |
| 9 | ANEMLL | Sent | 2026-03-17 | Awaiting | github.com/Anemll/Anemll/issues/49 |
| 10 | Performance bloggers | Backlog | - | - | Batch after initial responses |

---

## Wave 2 Targets — From Viral Tweet Engagers + ANE Community (March 2026)

Identified from community engagement with the ANE/Espresso viral content and the maderix ANE training breakthrough wave.

### Priority A: Direct ANE Researchers

#### 11. mechramc / Ramchand Kumaresan
- **Why**: Author of Orion (arxiv 2603.06728) — first open end-to-end ANE LLM training+inference system. Building on top of maderix's work. Closest peer in the space. Our inference-focused benchmarks complement their training work.
- **Channel**: GitHub @mechramc / github.com/mechramc/Orion — open an issue or discussion
- **Angle**: "Espresso reaches 519 tok/s / 3.41x CoreML for pure inference — would love to compare approaches and cross-reference benchmarks. Possible joint benchmark or citation?"
- **Ask**: Benchmark comparison, mutual citation, potential collaboration on inference side of Orion
- **Priority**: CRITICAL — most technically relevant new peer

#### 12. InsiderLLM.com
- **Why**: Published "Apple Neural Engine for LLM Inference: What Actually Works" — a high-quality guide with strong SEO that references the exact problem Espresso solves. Adding Espresso's benchmarks to this guide would be high-value placement.
- **Channel**: Contact form / Twitter if available
- **Angle**: "You wrote the definitive guide on ANE LLM inference. Espresso is the open-source answer to your question — happy to share benchmarks and a guest section."
- **Ask**: Add Espresso to their guide as a benchmark reference, guest post contribution
- **Priority**: HIGH — high SEO authority in the exact niche

### Priority B: Twitter/X Amplifiers

#### 13. @ronaldmannak (Ronald Mannak)
- **Why**: iOS/Swift developer, tweeted about ANE reverse engineering and CoreML 2-4x overhead. Direct overlap with Espresso's value prop. Developer credibility.
- **Channel**: Twitter/X @ronaldmannak
- **Angle**: "You posted about CoreML 2-4x overhead — Espresso eliminates it, 4.76x faster. Here's how: [repo link]"
- **Ask**: Retweet, star, feedback on API design
- **Priority**: HIGH — developer with credibility, right audience

#### 14. @LiorOnAI (Lior Alexander)
- **Why**: ~45K followers, AI educator who posted about the ANE training breakthrough. Audience is exactly the developer/ML community that would star Espresso.
- **Channel**: Twitter/X @LiorOnAI
- **Angle**: "You covered the ANE training breakthrough. The inference side has been solved too — Espresso: 4.76x faster, pure Swift, MIT. Worth a thread?"
- **Ask**: Tweet/thread about Espresso, star the repo
- **Priority**: HIGH — reach + right audience

#### 15. @BrianRoemmele
- **Why**: 1.8M+ followers, amplified the ANE training story massively. A single repost of Espresso would drive significant star momentum.
- **Channel**: Twitter/X @BrianRoemmele
- **Angle**: Short, punchy: "The ANE inference side of the story: Espresso hits 519 tok/s, 4.76x over CoreML, pure Swift. [repo]"
- **Ask**: Repost to his audience
- **Priority**: MEDIUM-HIGH — low personalization ceiling but huge reach multiplier

#### 16. @rohanpaul_ai (Rohan Paul)
- **Why**: Large ML/AI educator following (~50K+), covered M5 Neural Engine launch. Would connect Espresso to Apple Silicon hardware news.
- **Channel**: Twitter/X @rohanpaul_ai
- **Angle**: "You covered M5 Neural Engine — Espresso is the open-source framework that squeezes max inference performance from M-series ANE. Benchmark thread?"
- **Ask**: Feature in thread or newsletter
- **Priority**: MEDIUM

### Priority C: GitHub Projects (Integration Partners)

#### 17. anentropic / hft2ane
- **Why**: Tool for converting HuggingFace transformer models to ANE-accelerated versions. Direct integration: Espresso could be the inference runtime for their converted models.
- **Channel**: GitHub github.com/anentropic/hft2ane — open issue
- **Angle**: "Espresso could serve as the inference runtime for hft2ane-converted models — pure Swift, 4.76x faster than CoreML. Interested in exploring integration?"
- **Ask**: Integration discussion, README link
- **Priority**: HIGH — direct technical synergy

#### 18. FluidInference/FluidAudio
- **Why**: Swift SDK for local audio AI on Apple Neural Engine. Using CoreML today. Espresso's direct ANE approach could significantly improve their throughput.
- **Channel**: GitHub github.com/FluidInference/FluidAudio — open issue
- **Angle**: "Espresso achieves 4.76x faster inference than CoreML via direct ANE access in pure Swift — potential backend for FluidAudio?"
- **Ask**: Technical discussion, integration evaluation
- **Priority**: MEDIUM-HIGH — concrete integration opportunity

#### 19. llama.cpp ANE discussion thread (#336)
- **Why**: Active open GitHub discussion requesting ANE support for llama.cpp. Hundreds of developers watching this thread. Contributing Espresso's approach exposes it to the world's largest LLM inference project.
- **Channel**: github.com/ggml-org/llama.cpp/discussions/336
- **Angle**: "Espresso demonstrates a working pure-Swift ANE direct inference path at 519 tok/s. Sharing our MIL-level approach in case it's useful for llama.cpp's ANE work."
- **Ask**: Discussion contribution, potential integration collaboration with ggml team
- **Priority**: HIGH — massive audience, authoritative thread

#### 20. Code Coup / Coding Nexus (Medium)
- **Why**: Published two viral Medium articles about ANE training ("I Trained an LLM..." + "Someone Trained...") with tens of thousands of reads. The inference story is the natural follow-up piece.
- **Channel**: Medium @CodeCoup / contact via publication
- **Angle**: "You told the training side of the ANE story. Espresso is the inference chapter — 519 tok/s, 3.41x over CoreML, pure Swift. Interested in a follow-up piece or co-authorship?"
- **Ask**: Feature article, co-authored post, or a reference in their next ANE piece
- **Priority**: HIGH — proven ANE content reach, same audience

---

| # | Target | Status | Date Contacted | Response | Notes |
|---|--------|--------|---------------|----------|-------|
| 11 | mechramc/Orion | Todo | - | - | GitHub issue on mechramc/Orion |
| 12 | InsiderLLM.com | Todo | - | - | Contact via site + Twitter |
| 13 | @ronaldmannak | Todo | - | - | Twitter DM |
| 14 | @LiorOnAI | Todo | - | - | Twitter DM |
| 15 | @BrianRoemmele | Todo | - | - | Twitter reply/DM |
| 16 | @rohanpaul_ai | Todo | - | - | Twitter DM |
| 17 | anentropic/hft2ane | Todo | - | - | GitHub issue |
| 18 | FluidInference/FluidAudio | Todo | - | - | GitHub issue |
| 19 | llama.cpp ANE discussion | Todo | - | - | github.com/ggml-org/llama.cpp/discussions/336 |
| 20 | Code Coup / Coding Nexus | Todo | - | - | Medium contact |
