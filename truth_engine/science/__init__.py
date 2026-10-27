"""Scientific evidence sources (PubMed, OpenAlex) and study-quality grading."""
from .study_design import DESIGN_WEIGHT, StudyProfile, classify

__all__ = ["DESIGN_WEIGHT", "StudyProfile", "classify"]
