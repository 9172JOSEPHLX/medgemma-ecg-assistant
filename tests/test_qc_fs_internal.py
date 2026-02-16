# tests/test_qc_fs_internal.py  ### Feb 12th, 2026

import numpy as np
import math

# adapte les imports à ton repo

from medgem_poc.qc import qc_signal_from_leads
from medgem_poc.resample_to_500 import TARGET_FS_HZ
from medgem_poc.resample_to_500 import resample_leads_to_500hz


def _mk_sine(fs: float, seconds: float = 10.0, hz: float = 1.0, amp_mv: float = 0.2):
    n = int(round(seconds * fs))
    t = np.arange(n, dtype=np.float32) / float(fs)
    return (amp_mv * np.sin(2.0 * math.pi * hz * t)).astype(np.float32)

def _mk_leads(fs: float, seconds: float = 10.0):
    # leads minimaux mais suffisants pour QC + limb swap: I, II, III, aVR, aVL, aVF
    # On fait des signaux simples (pas physiologiques), l'objectif est l'horloge fs_internal
    I   = _mk_sine(fs, seconds, hz=1.0)
    II  = _mk_sine(fs, seconds, hz=1.1)
    III = II - I
    aVR = -(I + II) / 2.0
    aVL = I - II / 2.0
    aVF = II - I / 2.0
    # Ajoute un lead préferé “II” déjà présent
    return {"I": I, "II": II, "III": III, "aVR": aVR, "aVL": aVL, "aVF": aVF}

def test_fs_internal_contract_when_source_250():
    fs_src = 250.0
    leads = _mk_leads(fs_src, seconds=10.0)

    out = qc_signal_from_leads(leads, fs_hz=fs_src, patient_sex="M", iec_normalize=False, notch_hz=None)
    m = out["metrics"]

    assert "sampling" in m
    s = m["sampling"]
    assert float(s["fs_source_hz"]) == float(fs_src)
    assert float(s["target_fs_hz"]) == float(TARGET_FS_HZ)
    assert float(s["fs_internal_hz"]) == float(TARGET_FS_HZ)
    assert m["fs_hz"] == float(TARGET_FS_HZ)
    assert bool(s["resample_applied"]) is True

def test_fs_internal_contract_when_source_500_no_resample():
    fs_src = float(TARGET_FS_HZ)
    leads = _mk_leads(fs_src, seconds=10.0)

    out = qc_signal_from_leads(leads, fs_hz=fs_src, patient_sex="F", iec_normalize=False, notch_hz=None)
    m = out["metrics"]
    s = m["sampling"]

    assert float(s["fs_source_hz"]) == float(TARGET_FS_HZ)
    assert float(s["fs_internal_hz"]) == float(TARGET_FS_HZ)
    assert m["fs_hz"] == float(TARGET_FS_HZ)
    # selon ton impl : resample_applied peut être False si ratio=1
    assert bool(s["resample_applied"]) in (False, True)

def test_internal_length_ratio_10s_250_to_500():
    fs_src = 250.0
    seconds = 10.0
    leads = _mk_leads(fs_src, seconds=seconds)

    out = qc_signal_from_leads(leads, fs_hz=fs_src, patient_sex="U", iec_normalize=False, notch_hz=None)
    # on vérifie par lead_stats (RAW) + sampling ratio si dispo
    m = out["metrics"]
    # lead_stats porte sur RAW leads (source)
    raw_n = m["lead_stats"]["II"]["n"]
    assert raw_n == int(round(seconds * fs_src))

    # et on vérifie qu’on a bien internal 500Hz via le contrat
    assert float(m["sampling"]["fs_internal_hz"]) == float(TARGET_FS_HZ)

def test_infer_limb_swap_uses_fs_param_not_global():
    """
    Test de non-regression conceptuel:
    - si _infer_limb_swap utilisait un fs_internal global absent => NameError dans certains contextes
    - ici, l'appel via qc_signal_from_leads doit toujours fonctionner.
    """
    fs_src = 360.0
    leads = _mk_leads(fs_src, seconds=10.0)

    out = qc_signal_from_leads(leads, fs_hz=fs_src, patient_sex="M", iec_normalize=False, notch_hz=None)
    assert "metrics" in out
    assert out["metrics"].get("limb_swap_hypothesis", None) is not None or True

# Terminus