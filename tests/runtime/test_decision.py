import numpy as np

from hiad.runtime.decision import (
    classify_score,
    component_statistics,
    image_score_from_statistics,
    top_k_map_score,
)


def test_component_statistics_reports_compact_strongest_component_summary():
    anomaly_map = np.zeros((6, 6), dtype=np.float32)
    anomaly_map[0, 0] = 0.95
    anomaly_map[3:5, 3:5] = 0.8

    summary = component_statistics(anomaly_map, 0.5)

    assert summary["component_count"] == 2
    assert summary["anomalous_pixel_count"] == 5
    assert summary["largest_component_area"] == 4
    assert summary["strongest_component"]["area"] == 4
    assert np.isclose(summary["strongest_component"]["mean_score"], 0.8)
    assert "components" not in summary


def test_empty_component_summary_uses_finite_fallback_score():
    anomaly_map = np.zeros((2, 2), dtype=np.float32)

    summary = component_statistics(anomaly_map, 0.5)
    score = image_score_from_statistics(summary, fallback_score=0.42)

    assert summary["component_count"] == 0
    assert score == 0.42
    assert np.isfinite(score)


def test_image_score_rewards_coherent_component_over_isolated_noise():
    isolated = np.zeros((5, 5), dtype=np.float32)
    isolated[2, 2] = 0.9
    coherent = np.zeros((5, 5), dtype=np.float32)
    coherent[1:4, 1:4] = 0.7

    coherent_score = image_score_from_statistics(
        component_statistics(coherent, 0.5), 0.0
    )
    isolated_score = image_score_from_statistics(
        component_statistics(isolated, 0.5), 0.0
    )

    assert coherent_score > isolated_score


def test_component_score_is_invariant_to_uniform_resolution_scaling():
    low_resolution = np.zeros((8, 8), dtype=np.float32)
    low_resolution[2:4, 2:4] = 0.8
    high_resolution = np.repeat(np.repeat(low_resolution, 2, axis=0), 2, axis=1)

    low_score = image_score_from_statistics(
        component_statistics(low_resolution, 0.5), 0.0
    )
    high_score = image_score_from_statistics(
        component_statistics(high_resolution, 0.5), 0.0
    )

    assert np.isclose(low_score, high_score)


def test_top_k_map_score_uses_final_refined_map_values():
    anomaly_map = np.asarray([[0.1, 0.2], [0.8, 0.9]], dtype=np.float32)

    assert np.isclose(top_k_map_score(anomaly_map, 2), 0.85)


def test_image_score_preserves_a_stronger_global_task_prior():
    anomaly_map = np.zeros((5, 5), dtype=np.float32)
    anomaly_map[2, 2] = 0.6

    score = image_score_from_statistics(
        component_statistics(anomaly_map, 0.5),
        fallback_score=0.9,
    )

    assert score == 0.9


def test_classify_score_has_binary_states_and_fails_safe_for_invalid_scores():
    assert classify_score(0.3, 0.5) == "OK"
    assert classify_score(0.5, 0.5) == "OK"
    assert classify_score(0.5001, 0.5) == "NG"
    assert classify_score(float("nan"), 0.5) == "NG"
