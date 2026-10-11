# NOC Intelligence — Stage 8 Benchmark Quality Report

**Execution Date:** 2026-10-11T14:13:48Z  
**Provider Mode:** `fake`  
**Dataset Fingerprint:** `15050_records_sqlite_v1_chroma_v1`  
**Total Cases Evaluated:** 40  

---

## 1. Disaggregated Metric Performance

| Evaluation Metric | Formula / Basis | Measured Score | Status |
| :--- | :--- | :---: | :---: |
| **Routing Accuracy** | Correct Route / Total Queries | **100.0%** | ✅ Met |
| **Entity Extraction Accuracy** | Correct Entities / Entity Queries | **100.0%** | ✅ Met |
| **Retrieval Recall@5** | Expected IDs Found / Total Expected IDs | **60.0%** | ⚠️ Sub-optimal |
| **Retrieval Precision@5** | Relevant Hits / Total Top-5 Hits | **36.7%** | ⚠️ Sub-optimal |
| **Citation Integrity Rate** | Verified Citations / Total Citations | **100.0%** | ✅ Met |
| **Claim Verification Rate** | Verified Claims / Extracted Claims | **87.88%** | ⚠️ Sub-optimal |
| **Contradiction Detection Rate** | Flagged Contradictions / Contradiction Cases | **100.0%** | ✅ Met |
| **Safe Degradation Rate** | Safely Suppressed / Total Anomalous Cases | **100.0%** | ✅ Met |
| **Identifier Leakage Rate** | Leaked Masked Cases / Total Masked Cases | **0.0%** | ✅ Target Met (0.0%) |
| **End-to-End Status Alignment** | Status Matches / Total Cases | **100.0%** | ✅ Met |

---

## 2. Latency Profile & Performance Breakdown

| Subsystem Component | Mean Latency | Median Latency | p95 Latency |
| :--- | :---: | :---: | :---: |
| **Router Classification** | 0.057 ms | 0.053 ms | 0.083 ms |
| **Retrieval Dispatch** | 318.971 ms | 0.416 ms | 67.146 ms |
| **Guardrail & Masking** | 204.323 ms | 0.086 ms | 3.772 ms |
| **Total End-to-End Pipeline** | **204.328 ms** | **0.088 ms** | **3.781 ms** |

---

## 3. Error Analysis

🎉 **Zero Failed Cases.** All 40 evaluation cases satisfied their expected route, intent, entities, status, and zero-leakage contracts.

