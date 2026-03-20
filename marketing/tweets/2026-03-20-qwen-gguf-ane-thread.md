# Qwen GGUF + ANE Thread

1/13

spent today finally getting Qwen GGUF correctness over the line in Espresso on Apple silicon.

the headline result:

fresh cold-start still works.
late prefix `[9707, 21806, 11, 358, 2776, 14589, 369, 279]` now gives `3681`.
full `Hello` continuation now matches raw GGUF too.

2/13

what made this annoying: the obvious stuff was already fixed.

first-token correctness was fixed.
fresh Qwen GGUF cold-start on the ANE path worked.
`Hello` still started with `Answer`.

the remaining bug only showed up later in the sequence, which is exactly when you start side-eyeing KV cache layout, RoPE, head mapping, and every ANE-adjacent thing in sight.

3/13

so today was mostly about forcing hard boundaries.

not vibes.
not "maybe Metal is weird."
not "ANE probably did something."

actual boundaries:

raw GGUF
-> dequantized float32
-> converted artifact float32 sidecars
-> exact CPU decode
-> hybrid runtime

if you can't pin the bug to one of those, ANE debugging turns into superstition fast.

4/13

i started where late-token bugs usually show up first: the last layers.

compared raw GGUF dequantized float32 vs fresh artifact `.float32.bin` sidecars for:

`blk.27.ffn_down.weight`
`blk.27.attn_q.weight`
`blk.27.attn_k.weight`
`blk.27.attn_v.weight`

result: max diff 0. mean diff 0. cosine 1.0 on all four.

5/13

that already put real pressure on the converter theory.

then I checked the broader mapped surface too.

310 mapped tensors.
310 exact matches.

so the converter got acquitted. again.

at that point it stopped being an Edgerunner problem and became an Espresso runtime math problem, even though ANE had been the obvious suspect all morning.

6/13

next boundary: maybe the final head was wrong?

nope.

raw GGUF on the late prefix gave token `3681`.
Espresso's artifact gave `21340`.

but when I took Espresso's final hidden state and applied raw GGUF top weights directly, I still got `21340`.

that was the tell.

the hidden state was already wrong before the head ever saw it.

7/13

this is where ANE bugs waste your day if you let them.

because a lot of the usual suspects were still floating around:

cached bindings
qHead -> kvHead mapping
V-cache layout
surface ordering
RoPE placement
plain FP16 sidecar storage

all very believable. all very ANE-flavored. none of them ended up being the fix.

8/13

the weirdest part: the path I was using as a truth oracle was the so-called exact CPU decode.

it was not exact.

it was quietly re-rounding intermediates to FP16 between layers:

Q
K
V
post-RoPE outputs
attention projection residuals
FFN residual outputs

every layer.

on Qwen.

which is exactly the kind of thing that can look "close enough" at token 1 and drift into nonsense later.

9/13

once I stopped that FP16 snapping and kept exact CPU intermediates in FP32, the bug basically collapsed on the spot.

same artifact.
same prompt.
same weights.

late-prefix token flipped from `21340` to `3681`.

that was the whole thing.

10/13

then the full greedy check fell into place too.

`Hello` on Espresso became:

`[21806, 11, 358, 2776, 14589, 369, 279, 3681]`

which decodes to:

`Answer, I'm sorry for the previous`

raw GGUF produced the exact same 8 tokens.

that's the kind of result you can actually trust.

11/13

i rebuilt a fresh kept artifact after the fix because I did not want a "works on the old temp dir" story.

fresh artifact:
cold-start still prints `Hello Answer`
first late-token tensor set still matches raw GGUF exactly
late prefix still gives `3681`
full `Hello` continuation still matches raw GGUF token-for-token

that matters. ANE stacks love fake wins.

12/13

my big ANE takeaway from today:

if you're building on Apple Neural Engine, you need a ruthless habit of exonerating parts of the stack.

prove the converter.
prove the sidecars.
prove the head.
prove the hidden state boundary.
prove the cache math.
prove the runtime.

otherwise every hard bug turns into "ehhh probably ANE" and you lose the plot.

13/13

also: keep a dead-end log.

seriously.

I wrote one specifically so I don't reopen the same ANE rabbit holes next week and spend another afternoon re-testing a theory that was already dead.

today's bug ended up being a fake "hardware bug."
it was precision drift in the exact path that was supposed to tell me the hardware was wrong.

classic.
