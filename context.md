# MoE Interpretability Research: Technical Review and Context

> **Review Authored by**: Claude Opus 4.5 via Claude Code
> **Review Date**: December 26, 2025
> **Repository**: moe-interp

---

## Table of Contents

1. [Repository Context and Research Goals](#1-repository-context-and-research-goals)
2. [Critical Technical Gaps](#2-critical-technical-gaps)
3. [Bugs and Runtime Errors](#3-bugs-and-runtime-errors)
4. [Methodological Concerns](#4-methodological-concerns)
5. [Statistical and Experimental Issues](#5-statistical-and-experimental-issues)
6. [Code Quality Issues](#6-code-quality-issues)
7. [Data Processing Concerns](#7-data-processing-concerns)
8. [Suggestions for Improvement](#8-suggestions-for-improvement)
9. [Summary of Files Reviewed](#9-summary-of-files-reviewed)

---

## 1. Repository Context and Research Goals

### 1.1 Research Overview

This repository contains code for interpreting **Mixture of Experts (MoE)** language models, specifically focusing on:

- **DeepSeek-MoE-16B-Base**: 64 routed experts + 2 shared experts, top-6 routing
- **OLMoE-1B-7B**: 64 experts, top-8 routing across 16 MoE layers
- **Qwen-MoE**: Extension experiments

### 1.2 Primary Research Questions

1. **Expert Specialization**: Do different experts specialize for different domains (code, English, French, math)?
2. **Routing Patterns**: How do routing weights evolve across layers?
3. **Expert Co-activation**: Which experts frequently activate together?
4. **MOE-Lens Analysis**: Can we interpret what each expert "predicts" via logit lens?
5. **Perplexity vs Top-K**: How does perplexity change when limiting to top-k experts?
6. **Cosine Similarity**: How similar are top-1 expert representations vs top-k combined?

### 1.3 Key Experiments Conducted

| Notebook | Purpose | Models |
|----------|---------|--------|
| `moe-gate.ipynb` | Router weight analysis, PCA, expert selection | DeepSeek-MoE |
| `moe-lens.ipynb` | Logit lens per-expert analysis | DeepSeek-MoE |
| `pplx.ipynb` | Perplexity vs top-k expert ablation | DeepSeek-MoE |
| `cosine-sim.ipynb` | Cosine similarity of expert outputs | DeepSeek-MoE |
| `pca.ipynb` | PCA visualization of router logits | OLMoE |
| `olmoe-gate.ipynb` | Router analysis for OLMoE | OLMoE |

---

## 2. Critical Technical Gaps

### 2.1 **Incorrect Shared Expert Decomposition** (CRITICAL)

**File**: `deepseek_logit_lens.ipynb`, `moe-lens.ipynb`

**Issue**: The shared expert hook attempts to split the intermediate activations into two separate experts:

```python
# From deepseek_logit_lens.ipynb - cell 4
expert0_act = act[..., :1408]
expert1_act = act[..., 1408:]

# Then applies padding:
expert0_out = module.down_proj(
    F.pad(expert0_act, (0, 1408))  # Pad to match full width
)
```

**Problem**: This approach is **fundamentally flawed**:
1. DeepSeek's shared experts are **not** simply split halves of the hidden dimension
2. The padding approach (`F.pad`) creates artificial zeros that corrupt the down projection
3. The `n_shared_experts=2` configuration refers to 2 separate expert modules, not a split dimension
4. The correct approach would be to access `model.model.layers[i].mlp.shared_experts` as a module list

**Impact**: All shared expert analysis results are invalid and should not be trusted.

### 2.2 **Hardcoded Dimension Values** (HIGH)

**Files**: Multiple notebooks

```python
# Hardcoded dimensions that will break with different models
expert0_act = act[..., :1408]  # Assumes intermediate_size = 2816
expert1_act = act[..., 1408:]
```

**Issue**: The code assumes specific model dimensions (1408, 2816) rather than querying `model.config.intermediate_size`. This will silently produce incorrect results for other model sizes.

### 2.3 **Memory Leak in Hook Management** (MEDIUM)

**File**: `moe-lens.ipynb`, `deepseek_logit_lens.ipynb`

```python
def analyze_text(self, input_ids: torch.Tensor) -> dict:
    # ...
    final_hook = self.model.lm_head.register_forward_hook(get_final_output_hook)
    self.hook_handles.append(final_hook)
```

**Issue**: A new hook is registered on every call to `analyze_text()`, but only removed in `remove_hooks()`. If `analyze_text()` is called multiple times before cleanup, hooks accumulate.

### 2.4 **Router Weight Interpretation Confusion** (HIGH)

**File**: `moe-gate.ipynb`

The code conflates:
1. **Router logits** (pre-softmax scores)
2. **Router weights/probabilities** (post-softmax)
3. **Top-k selection** (binary selection of which experts)

In several places, `topk_weight` is treated as a probability when it's actually the post-softmax weight only for selected experts, which doesn't sum to 1 unless you consider only those experts.

---

## 3. Bugs and Runtime Errors

### 3.1 **IndexError in deepseek_logit_lens.ipynb** (CONFIRMED BUG)

**File**: `deepseek_logit_lens.ipynb`, cell 10

```python
for i in range(20):  # Iterates 0-19
    x = outputs.logits[0, i]  # But sequence length is only 15
```

**Error**: `IndexError: index 15 is out of bounds for dimension 1 with size 15`

**Status**: This error is visible in the notebook output, indicating the cell crashed during execution.

### 3.2 **Potential Division by Zero** (pplx.ipynb)

```python
# In perplexity calculation
log_prob = torch.log(prob)  # prob could be 0 for OOV tokens
```

**Issue**: No epsilon added to prevent log(0) = -inf. Should use `torch.log(prob + 1e-10)` or `torch.clamp(prob, min=1e-10)`.

### 3.3 **Unhandled Empty Expert Outputs** (moe-lens.ipynb)

```python
if info and info['tokens']:
    weight = info['weight']
    # ...
```

**Issue**: If `info['tokens']` is an empty list `[]`, it evaluates to `False`, but the code later assumes non-empty. However, `info['tokens'][0]` would fail on empty list before this check in some code paths.

### 3.4 **Device Mismatch Issues** (pca.ipynb)

```python
router_probs = router_probs.cuda()  # Forces CUDA
zeroed_probs = torch.zeros_like(router_probs, device='cuda')
```

**Issue**: The functions `get_last_token_router_probs` and `topk` force CUDA usage even though `setup_device()` correctly handles CPU fallback. The notebook output shows "CUDA not available, using CPU" but then these functions would crash.

---

## 4. Methodological Concerns

### 4.1 **Last-Token Bias in Analysis**

**Files**: `pca.ipynb`, `moe-gate.ipynb`, `olmoe-gate.ipynb`

Most analyses focus exclusively on the **last token**:

```python
last_token_logits = layer_logits[seq_len-1]
```

**Concern**: This introduces significant bias:
1. The last token has access to all previous context (causal attention)
2. Routing patterns differ substantially for earlier tokens
3. For domain classification, middle tokens may be more representative
4. Results may not generalize to inference-time behavior across sequences

**Recommendation**: Include analysis across all token positions, or at minimum stratified sampling.

### 4.2 **Lack of Statistical Significance Testing**

No notebooks include:
- Confidence intervals on reported metrics
- Statistical significance tests for domain differences
- Effect size calculations
- Multiple comparison corrections

For example, claiming "Expert 17 specializes in English" without p-values or bootstrap confidence intervals is not rigorous.

### 4.3 **No Control Experiments**

Missing controls:
1. **Random baseline**: What would random expert selection look like?
2. **Dense model comparison**: How do routing patterns compare to attention patterns in dense models?
3. **Permutation tests**: Are observed domain specializations significant vs permuted labels?

### 4.4 **Cosine Similarity Methodology Issues** (cosine-sim.ipynb)

```python
# The analysis compares top-1 expert output vs combined top-k output
# But this conflates several effects:
# 1. The top-1 expert alone
# 2. The weighted combination of all experts
# 3. Potential cancellation effects between experts
```

**Issue**: High cosine similarity between top-1 and combined output could mean:
- Top-1 dominates (intended interpretation)
- Other experts have low weights (trivial)
- Other experts have similar outputs (different interpretation)

Without disentangling these, conclusions are ambiguous.

---

## 5. Statistical and Experimental Issues

### 5.1 **Small and Imbalanced Datasets**

**File**: `data-ext/data-prep.ipynb`

| Domain | Sample Size | Token Count |
|--------|-------------|-------------|
| Code | 200 prompts | ~385K tokens |
| English | 200 prompts | variable |
| French | 200 prompts | ~290K tokens |
| Chinese | 275 entries | ~310K tokens |
| ArXiv | 25 articles | ~197K tokens |
| GSM8K | ~1300 problems | ~252K tokens |

**Issues**:
1. Very small sample sizes (200-275 examples per domain)
2. Token counts vary dramatically between domains
3. ArXiv has only 25 samples, far too few for robust conclusions

### 5.2 **Domain Confounds**

The domains differ in multiple confounded ways:
- **Length**: Code tends to be longer
- **Vocabulary**: Math uses special symbols
- **Structure**: Code has specific syntax patterns

Expert specialization could reflect any of these confounds, not semantic domain knowledge.

### 5.3 **Normalization Inconsistencies** (plot_pplx.ipynb)

```python
# Normalized to 0-1 range per dataset
normalized_perplexity = [(x - min_perp) / (max_perp - min_perp) for x in perplexity_values]
```

**Issue**: Each dataset is normalized independently, making cross-dataset comparisons meaningless. A "0.5 normalized perplexity" for GitHub code cannot be compared to "0.5" for English.

### 5.4 **Missing Perplexity Baselines**

The perplexity experiments show how performance degrades with fewer experts, but don't include:
1. Full model (all experts) baseline perplexity
2. Random expert selection baseline
3. Worst-k expert selection for comparison

---

## 6. Code Quality Issues

### 6.1 **Duplicate Imports**

```python
# deepseek_logit_lens.ipynb, cell 0
from pathlib import Path
from pathlib import Path  # Duplicate
```

### 6.2 **Commented-Out Code Pollution**

Many notebooks contain large blocks of commented code that should be removed or moved to separate exploration files. Examples:
- `pca.ipynb`: Multiple commented function calls
- `moe-gate.ipynb`: Commented analysis sections

### 6.3 **Inconsistent Variable Naming**

```python
# Sometimes using full words, sometimes abbreviations
router_logits_list  # Full
topk_idx  # Abbreviated
topk_weight  # Abbreviated
expert_indices  # Full
```

### 6.4 **Magic Numbers Without Constants**

```python
top_tokens = torch.topk(logits, k=5, dim=-1)  # Why 5?
batch_size = 8  # Why 8?
max_seq_len = min(4096, ...)  # Why 4096?
```

These should be defined as named constants with documentation.

### 6.5 **No Type Hints**

None of the main analysis functions include type hints, making the code harder to understand and maintain:

```python
# Current:
def get_moe_metadata(model, input_ids):

# Should be:
def get_moe_metadata(model: AutoModelForCausalLM, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
```

---

## 7. Data Processing Concerns

### 7.1 **Data Leakage Risk**

**File**: `data-ext/data-prep.ipynb`

The data preparation combines multiple sources without clear train/test splits. If the models being analyzed were trained on any of these sources, results would be confounded.

### 7.2 **Encoding Issues**

```python
with open(output_file, "w", encoding="utf-8") as f:
```

While UTF-8 encoding is used, there's no validation that the source data is valid UTF-8, which could cause silent corruption for Chinese or special characters.

### 7.3 **Token Count Verification Missing**

The code computes token counts but doesn't verify they match expectations or flag anomalies:

```python
print(len(tokens))  # Just prints, doesn't validate
```

### 7.4 **No Data Validation**

No checks for:
- Empty or whitespace-only prompts
- Excessively long prompts that get truncated
- Duplicate prompts
- Prompts with special tokens that might confuse the tokenizer

---

## 8. Suggestions for Improvement

### 8.1 High Priority Fixes

1. **Fix Shared Expert Analysis**: Completely rewrite the shared expert hook to correctly access DeepSeek's shared expert modules, not split dimensions.

2. **Add Statistical Rigor**:
   ```python
   from scipy import stats
   # Add confidence intervals
   ci = stats.bootstrap((data,), np.mean, confidence_level=0.95)
   ```

3. **Increase Sample Sizes**: Aim for at least 1000 samples per domain for robust statistics.

4. **Add Baseline Comparisons**: Include random routing and full-model baselines.

### 8.2 Medium Priority Improvements

5. **Create Configuration Files**: Move all magic numbers to a config:
   ```python
   # config.py
   TOP_K_TOKENS = 5
   BATCH_SIZE = 8
   MAX_SEQ_LEN = 4096
   ```

6. **Add Type Hints and Docstrings**: Improve code maintainability.

7. **Implement Proper Logging**: Replace print statements with logging:
   ```python
   import logging
   logger = logging.getLogger(__name__)
   logger.info(f"Processing {len(prompts)} prompts")
   ```

8. **Add Unit Tests**: Create tests for critical functions like router weight extraction.

### 8.3 Experimental Design Improvements

9. **Control for Confounds**:
   - Match sequence lengths across domains
   - Use vocabulary overlap analysis
   - Control for syntactic patterns

10. **Cross-Validation**: Use k-fold cross-validation for any domain classification claims.

11. **Ablation Studies**: Systematically ablate components to understand contributions.

12. **All-Token Analysis**: Don't limit to last-token; analyze full sequences.

### 8.4 Documentation Improvements

13. **Add Requirements File**: No `requirements.txt` or `pyproject.toml` found.

14. **Document Model-Specific Assumptions**: Many assumptions are DeepSeek-specific but not documented.

15. **Create Analysis Pipeline**: Convert notebooks to reproducible scripts.

---

## 9. Summary of Files Reviewed

### Main Notebooks

| File | Status | Critical Issues |
|------|--------|-----------------|
| `moe-lens.ipynb` | Analyzed | Shared expert decomposition flawed |
| `moe-gate.ipynb` | Analyzed | Router weight interpretation unclear |
| `pplx.ipynb` | Analyzed | No baselines, potential div-by-zero |
| `cosine-sim.ipynb` | Analyzed | Methodology ambiguous |
| `pca.ipynb` | Analyzed | Device mismatch, last-token bias |
| `deepseek_logit_lens.ipynb` | Analyzed | Runtime error, flawed shared expert analysis |
| `moe-lens-chat.ipynb` | Analyzed | Extension of main analysis |
| `moe-lens-visualize.ipynb` | Analyzed | Visualization notebook |
| `plot_pplx.ipynb` | Analyzed | Normalization issues |
| `olmoe-gate.ipynb` | Analyzed | Model-specific extension |

### Extension Notebooks

| File | Status |
|------|--------|
| `moe-gate-ext-qwen.ipynb` | Reviewed |
| `moe-gate-ext-deepseek.ipynb` | Reviewed |
| `moe-gate-ext-olmoe.ipynb` | Reviewed |

### Archive

| File | Status | Notes |
|------|--------|-------|
| `swap-zero-experts.ipynb` | Reviewed | Expert ablation experiments |
| `dense-model-exp.ipynb` | Reviewed | Dense model comparison |

### Data Preparation

| File | Status |
|------|--------|
| `data-ext/data-prep.ipynb` | Analyzed |

---

## Appendix: Severity Classification

| Severity | Count | Examples |
|----------|-------|----------|
| **CRITICAL** | 1 | Shared expert decomposition |
| **HIGH** | 4 | Hardcoded dimensions, router interpretation, device mismatch, methodology |
| **MEDIUM** | 6 | Memory leaks, empty checks, statistical issues |
| **LOW** | 10+ | Code quality, documentation |

---

## Conclusion

This research repository represents a substantial effort toward understanding MoE model internals. The visualizations and analysis frameworks are valuable contributions. However, several critical technical issues—particularly the flawed shared expert analysis and lack of statistical rigor—mean that **results should be interpreted with caution** until these issues are addressed.

The most impactful improvements would be:
1. Fixing the shared expert decomposition
2. Adding statistical significance testing
3. Increasing sample sizes
4. Including proper baselines and controls

---

*This review was conducted by Claude Opus 4.5 via Claude Code. All observations are based on static code analysis and do not include runtime verification of all code paths.*
