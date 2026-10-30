# Two Arc Pro B65 + Strata: what actually helps, measured

Field notes from running **Qwen3.8-Flash-Next-IQ3_S** on **two Intel Arc Pro B65 (32 GB each)** with Strata v0.2.8 / engine 0.1.54 (SYCL backend), 128 K context, layer-split across the two cards.

Everything below is measured on one machine, back to back, with the same client script. Negative results are included on purpose — most of the tuning knobs I tried were either useless or actively harmful, and that is the expensive part to discover.

## Hardware / software

| | |
|---|---|
| GPU | 2 × Intel Arc Pro B65, 32 GB, Battlemage (Xe2), xe kernel driver |
| CPU / board | AMD EPYC 7302, TYAN S8030 |
| Engine | Strata engine **0.1.54** (shipped with v0.2.8), SYCL, official binary |
| Model | Qwen3.8-Flash-Next-IQ3_S + MTP draft |
| Context | 131,072 |

## Working configuration

```
--spec 2 --max-context 131072 --prefill 4096 --kv-resident 4096
--pipeline-windows 2 --layer-split 25 --vram-reserve-mib 0
--kv int8 --stream-experts --expert-cache auto
env: ONEAPI_DEVICE_SELECTOR=level_zero:*   STRATA_STAGE_TRIM=1
```

Steady state: **12800 / 12800 experts resident** (100 %), VRAM 30.77 GiB on card 0 and 29.69 GiB on card 1.

## Headline numbers

| Metric | Result |
|---|---|
| Prompt read (20 K tokens, cold, steady) | **1,336 – 1,356 tok/s** |
| Decode (800 tok generation) | **57.1 – 57.8 tok/s** |
| MTP draft acceptance | 83 – 86 % |
| Cold prompt, first request after start | 588 – 929 tok/s (see the warm-up note) |

## What genuinely helped

### 1. `--spec 2` instead of `--spec 4` when running `--pipeline-windows 2`

This is the single most valuable change. With a pipelined width-2 draft window the engine answers as if `--spec 2` regardless, so a larger `--spec` only burns verification work.

| | `--spec 4` | `--spec 2` |
|---|---|---|
| Prefill (20 K cold) | 879.7 tok/s | **1,167 tok/s** (+33 %) |
| Decode (800 tok) | 55.1 tok/s | **59.2 tok/s** (+7.4 %) |
| MTP acceptance | 77.6 % | **88.6 %** (+11 pt) |

The +20 % quoted by upstream is for short chat contexts; at 128 K the win is smaller but the direction is the same.

### 2. Lowering `--max-context` from 262,144 to 131,072

Pinned KV host RAM dropped from **1.55 GiB to 0.77 GiB**, and the share of KV cells held resident doubled (7.8 % → 15.6 %). If your workload never exceeds 128 K, the extra context is pure pressure.

## Negative results — do not repeat these

### `--prefill 8192` is much slower on a layer-split pair

| `--prefill` | 20 K cold read |
|---|---|
| 4096 | **17.2 s** (1,167 tok/s) |
| 8192 | 35.7 s (561 tok/s) |

A large chunk leaves the second card of the split badly imbalanced. The upstream sweep that lists 4096 and 8192 as equally good does not hold across a layer split.

### `STRATA_KV_PREFETCH=1` crashes the engine

```
no progress for 60 s at layer 27 of the prompt chunk from token 8192
```

Exit code 139. The optimisation is validated on CUDA only; across two SYCL devices it does not survive. Taken out.

### `STRATA_ATTN_BATCH` is not worth touching

Tested 32 / 64 / 128, each with its own restart + warm-up + two 20 K cold reads:

| batch | prefill (steady) | decode | VRAM card 0 / card 1 |
|---|---|---|---|
| 32 | 1,336.4 tok/s | 57.3 / 57.7 | 30.77 / 29.69 GiB |
| 64 | 1,338.4 tok/s | 57.1 / 57.7 | 30.77 / 29.69 GiB |
| 128 | 1,356.4 tok/s | 57.2 / 57.8 | 30.77 / 29.69 GiB |

- 128 is only **1.5 %** faster than 32 — consistent with the 1.1–1.3 % upstream reports.
- **Decode is identical in all three**: the switch only affects the prompt path.
- **VRAM is identical in all three**: the attention-scratch difference is absorbed by `max({gdn_set_bytes, qsa_set_bytes, moe_set_bytes})`, and the MoE term is larger. So you cannot trade this knob for VRAM headroom.

### `LimitMEMLOCK=infinity` does nothing for the engine

The engine runs inside a container where the limit is already unlimited. Raising it on the host only affects the launcher.

### Three knobs that look like optimisations but do not apply

- `STRATA_MIRROR_MIB=0` — with 100 % expert residency the miss set is empty and the mirror is never allocated.
- `STRATA_CACHE_MESSAGE_BOUNDARY=1` — the source guards it with `!multi_gpu`; layer split is explicitly excluded.
- `STRATA_ARENA_PIN_GIB` — on Linux the whole arena is pinned by default (only Windows/WDDM caps it at 8 GiB).

## Measurement methodology (the part that changes conclusions)

Two effects are large enough to fabricate a result if you ignore them.

**1. The first request after start is 40–50 % slower.** Right after the engine comes up, a 20 K prompt reads at 588–929 tok/s; the same prompt once warm reaches 1,336+. Always send a throwaway request first, and only trust the third request onward.

**2. The same setting measured twice can differ by 12 %.** `STRATA_ATTN_BATCH=64` gave 1,194 tok/s on the first round and 1,338 on the second. I nearly published "batch 32 is 12 % faster than 64" on the strength of the first round alone.

So the bar for calling a difference real, in a restart-based A/B, is: **reproducible across two runs of the same value, and larger than the run-to-run drift (~12 % here)**. Anything under 5 % from a single round is noise.

Two practical rules that follow:

- Read the effective arguments from the engine's own log (`--spec`, `--prefill`, `--max-context` on the start line), not from the config file — the launcher may rewrite them.
- On the xe driver, VRAM is readable from `/sys/kernel/debug/dri/<BDF>/tile0/vram_mm`: the `usage:` field is bytes in use, and the `free=` on the `chunk_size:` line is what is left. `intel_gpu_top` fails with "Failed to detect engines" on xe.

## One caveat worth knowing if you run a single card

Engine 0.1.54 defaults `attn_batch` to 128 while its budget estimate (`bytes_needed_impl`) still counts 32, so roughly 1.2 GB of attention scratch is allocated outside the estimate. On a two-card split there is enough headroom that it never matters. On a **single 32 GB B65 with the standard config**, upstream reports the first request failing with `device buffers for a chunk of 512 tokens do not fit`. If you are single-card, set `STRATA_ATTN_BATCH=32` explicitly or leave more VRAM reserve.

## Files

- `benchmark-notes.md` — the raw per-run numbers behind the tables above.

Tested 2026-10-31. Engine 0.1.54, model IQ3_S, 128 K context, two cards.
