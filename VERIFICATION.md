# Deep verification report

## 1. BIBE decryption reliability (tiny preset)
- 16 random batches x 2 ciphertexts: **0 decryption failures**
- error magnitudes: max 2,613,215 = 0.0779 of q/4; mean 1,222,899
- extended-basis GS norms: max 666 (sigma1 = 4000, margin 6.01x)
- Lemma-31 selection produced an exact Lambda^perp basis in 0/16 batches (sublattice otherwise - documented deviation, AUDIT.md S3)
- wall time 96s

## 2. 1-D sampler statistics at scale
- sigma = 2^3.3: n=20000, mean/s = -0.0057 (tol 0.0283), std/s = 1.0057 (tol 5%)
- sigma = 2^8.2: n=20000, mean/s = +0.0036 (tol 0.0283), std/s = 0.9883 (tol 5%)
- sigma = 2^20.0: n=2000, mean/s = -0.0032 (tol 0.0894), std/s = 1.0155 (tol 5%)

## 3. TBIBE + GPV reliability (tiny threshold preset)
- 4 random (batch, act) threshold pre-decryptions: **0 failures**
- B_th = 1.5e+06 vs q/2 = 1.34e+08
- 4 threshold GPV sign/verify cycles: **0 failures**

## 4. Theorem 2/3/4 constraint system (provable parameters)
- lambda=80, ell=64: log2 q = 90, 9 constraints -> **all satisfied**
- lambda=128, ell=512: log2 q = 100, 9 constraints -> **all satisfied**
- lambda=128, ell=4096: log2 q = 105, 9 constraints -> **all satisfied**
- lambda=192, ell=1024: log2 q = 105, 9 constraints -> **all satisfied**
- threshold bounds (lambda=128, ell=512, 11-of-16): B_th = 2^141.6, q required >= 2^152.4

## 5. Toy-vs-provable gap (honesty check)
- tiny preset intentionally violates 2 provable constraints: ['sigma1 >= max(18 dim^1.5 log(dim lam) loglog q, 2 m^1.5 w(log(m+t)))', 'alpha < 1/((mu sigma1 + sqrt(m_c) sigma2) w(log mu))']
- (expected: toy parameters trade the security-proof constraints for runnability; correctness constraints all hold empirically, section 1)

