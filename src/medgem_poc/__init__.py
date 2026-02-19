# src/medgem_poc/__init__.py Feb 19th, 2026 version 16H29
"""
medgem_poc package (Kaggle/Jury-ready API surface)
"""

from .qc import qc_signal_from_leads
from .fhir_export import qc_to_fhir_observation

__all__ = [
    "qc_signal_from_leads",
    "qc_to_fhir_observation",
]

__version__ = "0.1.1"

