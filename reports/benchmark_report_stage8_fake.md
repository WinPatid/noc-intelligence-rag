# NOC Intelligence — Stage 8 Benchmark Quality Report (Refined)

**Execution Date:** 2026-10-11T17:15:45Z  
**Provider Mode:** `fake`  
**Dataset Fingerprint:** `15050_records_sqlite_v1_chroma_v1`  
**Total Cases Evaluated:** 40  

---

## 1. Disaggregated Metric Performance

| Evaluation Metric | Formula / Basis | Measured Score | Status |
| :--- | :--- | :---: | :---: |
| **Routing Accuracy** | Correct Route / Total Queries | **100.0%** | ✅ Met |
| **Entity Extraction Accuracy** | Correct Entities / Entity Queries | **100.0%** | ✅ Met |
| **Retrieval Recall@5** | Expected IDs Found / Total Expected IDs | **83.3%** | ⚠️ Baseline |
| **Retrieval Precision@5** | Relevant Hits / Total Top-5 Hits | **46.3%** | ⚠️ Baseline |
| **Citation Source-ID Validity** | Valid Retrieved Source IDs / Total Citations | **93.18%** | ⚠️ Unretrieved IDs Cited |
| **Citation Ground-Truth Support** | Ground-Truth Supported / Evaluable Citations | **77.5%** | ⚠️ Mismatched Citations |
| **Factual Claim Verification** | Verified Claims / Extracted Factual Claims | **98.13%** | ⚠️ Unverified Claims |
| **Adversarial False-Claim Rejection** | Flagged Conflicts / Adversarial Claims | **100.0%** | ✅ Met (100.0%) |
| **Unsupported Claim Rejection** | Flagged Unretrieved / Fabricated Claims | **100.0%** | ✅ Met (100.0%) |
| **Contradiction Detection Rate** | Flagged Contradictions / Contradiction Cases | **100.0%** | ✅ Met (100.0%) |
| **Safe Degradation Rate** | Safely Suppressed / Total Anomalous Cases | **100.0%** | ✅ Met (100.0%) |
| **Identifier Leakage Rate** | Leaked Masked Cases / Total Masked Cases | **0.0%** | ✅ Target Met (0.0%) |
| **End-to-End Status Alignment** | Status Matches / Total Cases | **97.5%** | ✅ Met |

---

## 2. Latency Profile & Performance Breakdown

* **Vector Store Cold-Start Loading:** `7876.49 ms` (SentenceTransformers + ChromaDB initial weight import)
* **Total Warm-Up Duration:** `7882.88 ms` across `4` queries
* **Steady-State Benchmark Samples:** `40` evaluation cases (cold-start model loading excluded)

| Subsystem Component | Mean Latency | Median Latency | p95 Latency |
| :--- | :---: | :---: | :---: |
| **Router Classification** | 0.061 ms | 0.051 ms | 0.123 ms |
| **Retrieval Dispatch** | 11.593 ms | 0.484 ms | 65.291 ms |
| **Guardrail & Masking** | 0.171 ms | 0.201 ms | 0.39 ms |
| **True Full End-to-End Pipeline** | **4.509 ms** | **0.452 ms** | **16.893 ms** |

---

## 3. Safe Presentation Contract & Privacy Boundary (Stage 9 Ready)

To prevent data leakage, user-facing client applications (e.g. Streamlit UI in Stage 9) must adhere to the following contract:

1. **Approved User-Facing Fields:**
   - `final_answer`: Sanitized natural language response with pseudonymized identifiers (`[TICKET-***001]`, `[CUSTOMER-0001]`).
   - `citations`: Verified reference tags with masked labels and IDs.
   - `warnings`: Masked operational notifications.
   - `is_safe_to_present`: Boolean safety flag. If `False`, response MUST be suppressed.

2. **Forbidden Internal Diagnostic Fields (Never Render in UI):**
   - `original_answer`: Contains raw unmasked operational data.
   - `claims`: Contains unmasked factual values and evidence values.
   - `contradictions` & `unverified_claims`: Contain raw operational identifiers in debug strings.
   - `masked_identifiers_map`: Contains the reverse pseudonymization map.

3. **Masking Scope & Known Entity Gaps:**
   - **Covered Entities:** Trouble Tickets (`INC-...`), Work Orders (`WO-...`), Circuits (`CKT-...`), Customer Names (Thai & English).
   - **Known Unmasked Gaps:** Technician aliases (`field_engineer_alias`), customer contact persons (`contact_person`), phone numbers (`contact_phone`), site locations (`site_name`), and IP addresses.

---

## 4. Error Analysis

Total Failed Cases: **1**

### BM-SQL-06 (sql_lookup)
- **Expected:** Route: SQL_LOOKUP, Intent: BRIDGE_WO_TO_TICKETS, Status: VERIFIED
- **Actual:** Route: SQL_LOOKUP, Intent: BRIDGE_WO_TO_TICKETS, Status: CONTRADICTED
- **Failed Subsystem:** Critic
- **Diagnosis:** Routing or status mismatch against ground truth

