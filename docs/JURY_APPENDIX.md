# Jury Appendix — Build Fingerprint & Proofs (MedGemma ECG Assistant)

This appendix exists because Kaggle WriteUps can be locked after the submission deadline.  
It provides an **immutable, auditable fingerprint** of the code executed on Kaggle and a fast checklist to validate the proof artifacts.

## 0) Source-of-truth policy (important)

- **Immutable code reference for judges:** Git tag **`v0.1.1-kaggle-jury-20260312`** (do not use `main` HEAD for proof).
- **Post-jury work:** must happen on a separate development branch (e.g. `dev/post-jury`) and **must not modify the tag**.

## Kaggle notebook & write-up

- Public proof run notebook: https://www.kaggle.com/code/josluizlunaxavier/notebook8c2face391  
- Kaggle WriteUp (official submission):  
  https://www.kaggle.com/competitions/med-gemma-impact-challenge/writeups/ecg-assistant-offline-qc-medgemma-structured  
- GitHub Release (immutable code reference for this appendix):  
  https://github.com/9172JOSEPHLX/medgemma-ecg-assistant/releases/tag/v0.1.1-kaggle-jury-20260312  
- Demo video (YouTube): https://youtu.be/Cu7NKj1g9lA

---

## 1) Immutable build fingerprint (Kaggle “working” import)

The notebook **copies the package from the dataset vNext folder into `/kaggle/working/`** and forces imports from there (to avoid legacy/flat imports from `/kaggle/input`).

**Package**
- `medgem_poc==0.1.1`

**Imported modules (Kaggle working)**
- `medgem_poc.qc`  
  File: `/kaggle/working/appli_medgem_poc_code/src/medgem_poc/qc.py`  
  SHA256: `c4fd64429491a3846de2451b1bf7b60640f1bcfb45f8d4bec7e9c9d37c87da38`  
  Signals: `detect_limb_reversal` present; `WARN_LIMB_SWAP_SUSPECT` enabled; **baseline preserved**

- `medgem_poc.edge_metrics`  
  File: `/kaggle/working/appli_medgem_poc_code/src/medgem_poc/edge_metrics.py`  
  SHA256: `0374680db9d0e6d082f94a2b1e47ca91c33a2381506362759b277b0cf6c3f234`  
  Signals: `EdgeMetricsCollector` present; `degradation_mode_effective` supported; **CPU VRAM fields are `null`**

### Tagged source files (immutable)

These links must be used for audit (the tag is frozen):

- qc.py (tag `v0.1.1-kaggle-jury-20260312`):  
  https://github.com/9172JOSEPHLX/medgemma-ecg-assistant/blob/v0.1.1-kaggle-jury-20260312/src/medgem_poc/qc.py

- edge_metrics.py (tag `v0.1.1-kaggle-jury-20260312`):  
  https://github.com/9172JOSEPHLX/medgemma-ecg-assistant/blob/v0.1.1-kaggle-jury-20260312/src/medgem_poc/edge_metrics.py

---

## 2) Dataset snapshot used to build the working copy (code3 vNext)

Dataset folder (vNext):
- `/kaggle/input/datasets/josluizlunaxavier/jllxavierappli-medgem-poc-code3/kaggle_code3_vNext`

Files & SHA256:
- `src/medgem_poc/qc.py`  
  SHA256: `c4fd64429491a3846de2451b1bf7b60640f1bcfb45f8d4bec7e9c9d37c87da38`

- `src/medgem_poc/edge_metrics.py`  
  SHA256: `0374680db9d0e6d082f94a2b1e47ca91c33a2381506362759b277b0cf6c3f234`

- `tools/bench_swap_detection.py`  
  SHA256: `a98e50adbb0d1db2bfc4b2b640f8c0fbfd41bf2af41551f5166a3bec727c9b4d`

- `tools/generate_swap_cases.py`  
  SHA256: `2a816583c0c6b79b0cd91d8293690a8fd6f721396fbf2ba6e60943825cdc0991`

- `README.txt`  
  SHA256: `41261846f096d587c298ba847ef38f9a116194dbdf19771cef0994babc362405`

---

## 3) Proof artifacts written by the notebook (Kaggle outputs)

After running the notebook end-to-end, the pipeline writes verifiable Edge Metrics JSON reports to:

- `/kaggle/working/outputs/edge_metrics_09487_ok.json`  
  Baseline **PASS/PASS/PASS** case (GPU inference if available). Contains latency + RAM/VRAM + resilience fields.

- `/kaggle/working/outputs/edge_metrics_09487_swap.json`  
  Swap case. QC emits **`WARN_LIMB_SWAP_SUSPECT`** as an **acquisition warning** (not a diagnosis; not a FAIL).

- `/kaggle/working/outputs/edge_metrics_11899.json`  
  Baseline **WARN/WARN/WARN** case (quality warning preserved).

- `/kaggle/working/outputs/edge_metrics_09487_ok_force_cpu.json`  
  Failure injection (`FORCE_CPU=1`): device is CPU, **VRAM is n/a**, fallback is used, and `degradation_mode_effective = QC_ONLY`.

---

## 4) 10-second checklist (what judges should verify)

Open any JSON proof artifact and check:

1) **Device**
   - runtime device is `cuda:*` (GPU) or `cpu` (failure injection)

2) **Latency**
   - `latency_ms.total` (and `pre`, `infer`, `post`)

3) **Memory**
   - `memory.ram_rss_mb`
   - VRAM fields are present only when GPU is used
   - When `device=cpu`, VRAM fields must be `null` / pretty prints as `n/a`

4) **Resilience**
   - `resilience.offline_mode`
   - `resilience.fallback_used`
   - `resilience.fallback_reason`
   - `resilience.degradation_mode_effective`

5) **QC acquisition warnings**
   - Swap case must include `WARN_LIMB_SWAP_SUSPECT` (**acquisition warning only**)

---

## 5) Runtime behavior (intended)

- **QC gate semantics:** `PASS / WARN / FAIL`  
  - `WARN` indicates acquisition quality concerns or resilience degradations.
  - `FAIL` is reserved for missing/invalid signal or severe issues (**not used solely for SWAP**).

- **SWAP policy:**  
  - Limb-electrode inversion is surfaced as **`WARN_LIMB_SWAP_SUSPECT`**.
  - This is an **acquisition** warning ("recheck electrode placement / reacquire ECG"), **not a clinical diagnosis**.

- **Failure injection policy:**  
  - Setting `FORCE_CPU=1` forces a degraded mode:
    - `device=cpu`
    - VRAM reported as `null` / pretty `n/a`
    - fallback fields set
    - `degradation_mode_effective = QC_ONLY`

---

## 6) Safety note

This project is a **demo of offline-first ECG ingestion + QC gating + structured reporting**.  
QC warnings are used for acquisition quality and resilience handling; it does **not** claim autonomous clinical diagnosis.

---

## 7) Links

- Kaggle WriteUp (official submission):  
  https://www.kaggle.com/competitions/med-gemma-impact-challenge/writeups/ecg-assistant-offline-qc-medgemma-structured

- Kaggle Notebook (public proof run):  
  https://www.kaggle.com/code/josluizlunaxavier/notebook8c2face391

- GitHub Release (immutable code reference for this appendix):  
  https://github.com/9172JOSEPHLX/medgemma-ecg-assistant/releases/tag/v0.1.1-kaggle-jury-20260312

- Demo video (YouTube): https://youtu.be/Cu7NKj1g9lA
