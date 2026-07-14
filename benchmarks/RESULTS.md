# BLT25 benchmark results

Times are wall-clock seconds (median), single-threaded Python/NumPy.
Toy parameters: NOT cryptographically secure (see PARAMS.md).

## BIBE (Section 5)

| preset | n | d | ell | log2 q | setup | enc | predec (cold/warm) | dec (cold/warm) | pk | ct | sbk | ell x ct | err/q4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tiny | 4 | 4 | 2 | 28 | 0.02s | 0.159s | 13.34/0.05s | 13.45/0.00s | 12 KiB | 4.2 KiB | 420 B | 8 KiB | 0.073 |
| small | 4 | 6 | 4 | 29 | 0.02s | 0.223s | 260.53/0.06s | 263.98/0.00s | 16 KiB | 5.3 KiB | 479 B | 21 KiB | 0.086 |

Identity tags are representative pseudorandom values.  About 17% of random label sets (measured at the tiny preset) make the Lemma-31 leading block singular and push that batch onto the ~10-20x slower rank-profile fallback (correctness unaffected) - visible as bimodal cold-derivation times across the tables below.


## Batch-size scaling (cold derivations)

| preset | ell' | predec | dec | sbk | batch ct |
|---|---|---|---|---|---|
| tiny | 1 | 0.30s | 0.25s | 546 B | 4 KiB |
| tiny | 2 | 12.36s | 12.23s | 546 B | 8 KiB |
| small | 1 | 0.53s | 0.48s | 566 B | 5 KiB |
| small | 2 | 2.10s | 2.08s | 566 B | 11 KiB |
| small | 3 | 5.47s | 5.35s | 566 B | 16 KiB |
| small | 4 | 11.72s | 11.30s | 566 B | 21 KiB |

## TBIBE (Section 6, two-round threshold pre-decryption)

| preset | N | tau | log2 q | setup | r1/party | r2 total | combine | dec | comm r1+r2/party | ||sbk|| / B_th |
|---|---|---|---|---|---|---|---|---|---|---|
| tiny | 4 | 3 | 29 | 0.10s | 0.004s | 1.12s | 0.000s | 0.00s | 1.2+1.2 KiB | 0.128 |
| small | 4 | 3 | 30 | 0.11s | 0.005s | 2.65s | 0.000s | 0.00s | 1.3+1.3 KiB | 0.137 |

## Threshold GPV signatures (Section 7)

| preset | N | tau | setup | sign r1 | sign r2 | combine | verify | sig |
|---|---|---|---|---|---|---|---|---|
| tiny | 4 | 3 | 0.10s | 0.013s | 0.004s | 0.000s | 0.0002s | 728 B |
| small | 4 | 3 | 0.11s | 0.013s | 0.004s | 0.000s | 0.0003s | 797 B |

## Theorem-2 (provable) parameter sizes, for reference

These are the sizes the paper's proofs require at lambda = 128; far beyond what this prototype executes.

| lambda | ell | log2 q | t | pk | ct | sbk | ell x ct |
|---|---|---|---|---|---|---|---|
| 128 | 64 | 95 | 3125120 | 4583 MiB | 37087 KiB | 71 KiB | 2318 MiB |
| 128 | 512 | 100 | 3289600 | 5078 MiB | 41094 KiB | 75 KiB | 20547 MiB |
| 128 | 4096 | 105 | 3454080 | 5599 MiB | 45306 KiB | 79 KiB | 181223 MiB |
