# Raw measurements

All runs: Qwen3.8-Flash-Next-IQ3_S, two Arc Pro B65, engine 0.1.54, 128 K context, layer split 25, `--pipeline-windows 2`, `--prefill 4096`.

Client script: one ~20 K-token prompt (cold read — the salt is at the very front so the prefix cache cannot hit) followed by an 800-token generation. Each configuration got its own restart, then a short warm-up request, then two of these round trips. The **B** column is the steady-state value; **A** is the first large prompt after warm-up, which is still contaminated by start-up cost.

## `--spec` comparison

| | prefill | decode 800 | MTP accepted |
|---|---|---|---|
| `--spec 4` | 879.7 tok/s | 55.1 tok/s | 361/465 = 77.6 % |
| `--spec 2` | 1,167.0 tok/s | 59.2 tok/s | 272/307 = 88.6 % |

Same 20,063-token prompt, 128 K context, both warm.

## `STRATA_ATTN_BATCH` sweep

| batch | round | A (first) | B (steady) | decode 800 | VRAM card 0 | VRAM card 1 |
|---|---|---|---|---|---|---|
| 32 | 1 | 854.6 | 1,336.4 | 57.3 / 57.7 | 30.77 GiB | 29.69 GiB |
| 64 | 1 | 819.6 | 1,194.1 | 57.2 / 57.7 | 30.76 GiB | 29.68 GiB |
| 64 | 2 | 922.8 | 1,338.4 | 57.1 / 57.7 | 30.77 GiB | 29.69 GiB |
| 128 | 1 | 929.0 | 1,356.4 | 57.2 / 57.8 | 30.77 GiB | 29.69 GiB |

Free VRAM at batch 32 / 64 / 128 was identical: 1,149 MiB on card 0 and 2,257 MiB on card 1 in all three cases.

Note the 64 rows: **1,194 vs 1,338 tok/s for the same value**, 12 % apart. That spread is the reason the 1.5 % gap between 32 and 128 cannot be claimed from a single round.

## `--prefill` chunk size

| `--prefill` | 20 K cold read | rate |
|---|---|---|
| 4096 | 17.2 s | 1,167 tok/s |
| 8192 | 35.7 s | 561 tok/s |

## Cold-start behaviour

| request after start | 20,063-token read |
|---|---|
| 1st (no warm-up) | 588 tok/s |
| 2nd | 1,167 tok/s |
| steady | 1,336 – 1,356 tok/s |

## MTP acceptance across the working config

| prompt | accepted / drafted |
|---|---|
| 800-token generation, technical prose | 261/311 = 83.9 % |
| 800-token generation, second run | 264/306 = 86.3 % |
| 800-token generation, third run | 267/310 = 86.1 % |

## How VRAM was read

```bash
sudo cat /sys/kernel/debug/dri/<BDF>/tile0/vram_mm
#   usage: <bytes in use>
#   chunk_size: 4KiB, total: ..., free: <bytes left>MiB
```

`intel_gpu_top` reports "Failed to detect engines" on the xe driver and `xpu-smi` was not installed on this machine, so the debugfs node above is the only reliable source.
