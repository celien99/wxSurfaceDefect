import numpy as np


def compute_reference_coverage(candidate_mask, reference_prior) -> float:
    candidate = np.asarray(candidate_mask)
    prior = np.asarray(reference_prior)
    if candidate.ndim != 2 or prior.ndim != 2:
        raise ValueError("Candidate and reference masks must be two-dimensional")
    if candidate.shape != prior.shape:
        raise ValueError("Candidate and reference mask shape must match")
    if candidate.dtype != np.bool_ or prior.dtype != np.bool_:
        raise TypeError("Candidate and reference masks must be boolean")
    prior_area = int(prior.sum())
    if prior_area == 0:
        raise ValueError("Reference prior must not be empty")
    intersection = int(np.logical_and(candidate, prior).sum())
    return intersection / prior_area
