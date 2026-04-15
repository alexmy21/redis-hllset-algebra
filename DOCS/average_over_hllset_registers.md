# Calculating HLLSet cardinality

**Setup**:

- **HLL**: each register stores only the **maximum** state observed.
- **Your case**: each register stores the **set** of all states observed (all trailing-zero counts that appeared for tokens landing in that register).

So at the end, for each register $( i )$, you know which states $( s )$ were observed at least once in that register.

Thus:

- You have an $( n \times m )$ binary matrix $( A )$ where $( A_{i,s} = 1 )$ if register $( i )$ observed state $( s )$ at least once.
- $( c_s = \sum_{i=1}^n A_{i,s} )$ = number of registers that observed state $( s )$ (at least once).
- Different states can have different $( c_s )$ values.

You want: **unbiased weighted average** where weights come from $( c_s )$ (or can be derived from it).

---

## Reformulating the statistical model

Let:

- $( N )$ = total number of tokens (unknown)
- $( n )$ = number of registers (known)
- $( p_{i,s} )$ = probability that a **single token** has register $( i )$ and state $( s )$

For a good hash:

```math
p_{i,s} = \frac{1}{n} \cdot (2^{-s} - 2^{-(s+1)}) = \frac{1}{n} \cdot 2^{-(s+1)}
```

for $( s = 0, 1, 2, \dots )$, independent across tokens.

Let $( N_{i,s} )$ = number of tokens that landed in register $( i )$ with state exactly $( s )$.

Then $( N_{i,s} \sim \text{Poisson}(N \cdot p_{i,s}) )$ approximately (or Binomial, but Poisson approximation fine for large N).

Register $( i )$ **observes** state $( s )$ if $( N_{i,s} \ge 1 )$.  
So $( A_{i,s} = 1 )$ with probability $( 1 - e^{-N p_{i,s}} )$.

Thus:

```math
E[c_s] = n \cdot \left(1 - e^{-N \cdot 2^{-(s+1)} / n}\right)
```

---

## Can you get unbiased weighted average from $( c_s )$?

Let $( Y_s )$ = some value associated with state $( s )$ (fixed per state, e.g., cost, measurement).

True weighted average (weighted by true frequency of each state):

```math
\mu = \frac{\sum_s (N \cdot 2^{-(s+1)}) \cdot Y_s}{\sum_s N \cdot 2^{-(s+1)}} = \frac{\sum_s 2^{-(s+1)} Y_s}{\sum_s 2^{-(s+1)}} = \frac{\sum_s 2^{-(s+1)} Y_s}{1}
```

(since $( \sum_{s=0}^\infty 2^{-(s+1)} = 1 )$).

So **if $( Y_s )$ is fixed per state**, $( \mu )$ is a known constant — no estimation needed, independent of data.

But if $( Y_s )$ varies per token (e.g., each token has a value, and you want weighted average by state frequency), then you'd need to know average $( Y )$ per state, which you can't get from $( c_s )$ alone — you'd need per-state sum of $( Y )$ across tokens.

---

## What if $( Y_s )$ is **not** fixed but you want to use $( c_s )$ as weights?

Suppose you define:

```math
\hat{\mu} = \frac{\sum_s c_s \cdot \bar{Y}_s}{\sum_s c_s}
```

where $( \bar{Y}_s )$ = average $( Y )$ for tokens with state $( s )$ (observed from data).

Then:

```math
E[\hat{\mu}] \approx \frac{\sum_s E[c_s] \cdot E[\bar{Y}_s]}{\sum_s E[c_s]}
```

But $( E[\bar{Y}_s] )$ = true average $( Y )$ for state $( s )$ if you observe all tokens with state $( s )$ across all registers. But $( c_s )$ weights registers, not tokens.

The problem: $( c_s )$ is # registers containing state $( s )$, which is **not** proportional to true frequency $( f_s = N \cdot 2^{-(s+1)} )$ unless $( N \cdot 2^{-(s+1)} / n )$ is small.

---

## Known solution approach

This is analogous to **estimating a weighted average from presence/absence data across sites**, where each site (register) has multiple sampling occasions (tokens).

The unbiased estimator uses **inverse probability weighting**:

Let $( \hat{p}_s = 1 - e^{-c_s / n} )$??? No — better:

We have:

```math
E[c_s] = n(1 - e^{-\lambda_s})
```

where $( \lambda_s = N \cdot 2^{-(s+1)} / n )$.

If you could estimate $( \lambda_s )$ from $( c_s )$, then $( \hat{f}_s = n \hat{\lambda}_s )$ would be unbiased for true frequency $( f_s )$.

Solve: $( c_s/n = 1 - e^{-\hat{\lambda}_s} )$ → $( \hat{\lambda}_s = -\ln(1 - c_s/n) )$.

Then $( \hat{f}_s = n \cdot \hat{\lambda}_s = -n \ln(1 - c_s/n) )$.

This is unbiased for $( f_s )$ when $( N )$ is large and Poisson approximation holds.

Thus unbiased weighted average:

```math
\hat{\mu} = \frac{\sum_s \hat{f}_s \cdot \bar{Y}_s}{\sum_s \hat{f}_s} = \frac{\sum_s [-n \ln(1 - c_s/n)] \cdot \bar{Y}_s}{\sum_s [-n \ln(1 - c_s/n)]}
```

This works if $( \bar{Y}_s )$ is the sample mean of $( Y )$ for all tokens with state $( s )$ (which you can compute if you record $( Y )$ per token and per state).

---

## Has someone solved this?

Yes — this is essentially a **Horvitz–Thompson estimator** for abundance-weighted means using presence/absence data with known per-occasion detection probability (here $( 2^{-(s+1)}/n )$ per token, but aggregated over $( N )$ tokens).

The specific formula $( \hat{f}_s = -n \ln(1 - c_s/n) )$ appears in **species richness estimation** (Chao1 estimator) and in **occupancy modeling** for abundance from detection/non-detection data when detection probability is known.

So the answer: **Yes, unbiased weighted average is possible** using:

```math
\hat{\mu} = \frac{\sum_s \left[-n \ln\left(1 - \frac{c_s}{n}\right)\right] \cdot \bar{Y}_s}{\sum_s \left[-n \ln\left(1 - \frac{c_s}{n}\right)\right]}
```

provided $( \bar{Y}_s )$ is computed from all tokens with state $( s )$.

---

## Implementation Notes (April 2026)

### Cardinality Estimation

The Horvitz-Thompson estimator is now implemented in both:

- **Cython**: `core/hll_core.pyx` → `cardinality_ht()`
- **Rust**: `redis_hllset/module/src/hllset.rs` → `cardinality_ht()`

Both use `cardinality()` as the main entry point, which delegates to `cardinality_ht()`.

### Saturation Handling

When $c_s = n$ (all registers have bit $s$ set), the formula $-n \ln(0)$ is undefined.

Solution: **geometric extrapolation** from the first non-saturated state.

Since state $s$ has expected frequency $2 \times$ state $s+1$:

- Find first non-saturated state $k$ with estimate $\hat{f}_k$
- For saturated state $s < k$: use $\hat{f}_s = \hat{f}_k \times 2^{(k-s)}$

### Accuracy Results

| Cardinality | HT Error | Old HLL Error |
|-------------|----------|---------------|
| 10          | 0.00%    | 0.00%         |
| 100         | 0.00%    | -1.00%        |
| 1,000       | +0.20%   | -3.00%        |
| 10,000      | +2.27%   | +2.11%        |
| 100,000     | -4.03%   | -66.36%       |

The HT estimator with saturation extrapolation provides:
- **Mean absolute error**: ~1%
- **Max absolute error**: ~4%

This is significantly better than the original HLL approach which failed badly at high cardinalities due to incorrect treatment of saturated states.
