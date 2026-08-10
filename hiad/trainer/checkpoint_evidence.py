import os

import torch
import torch.multiprocessing as mp

from hiad.detectors.config import detector_config_for_task
from hiad.preprocessing import ForegroundPreprocessorRegistry
from hiad.runtime.evidence import collect_task_evidence
from hiad.runtime.logging import create_logger
from hiad.runtime.partition import round_robin_partition
from hiad.runtime.randomness import seed_everything
from hiad.scoring import merge_worker_evidence


def _collect_checkpoint_evidence_in_device(
    gpu_id,
    detector_class,
    config,
    samples,
    tasks,
    batch_size,
    checkpoint_root,
    log_root,
    seed,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    seed_everything(seed)
    logger = create_logger(
        f"calibration_logger_device{gpu_id}",
        os.path.join(log_root, f"calibration_log_device{gpu_id}.log"),
    )
    preprocessing_config = getattr(config, "preprocessing", None)
    if preprocessing_config is None:
        raise ValueError("Calibration worker requires preprocessing config")

    preprocessors = ForegroundPreprocessorRegistry.from_checkpoint(
        str(checkpoint_root),
        device,
        runtime_config=preprocessing_config,
        logger=logger,
    )

    def detector_provider(task):
        task_name = task["name"]
        detector_config = detector_config_for_task(config, task)
        detector = detector_class(
            **detector_config,
            logger=logger,
            device=device,
            seed=seed,
        )
        checkpoint_path = os.path.join(
            checkpoint_root,
            f"{task_name}_weight.pt",
        )
        logger.info(f"Load checkpoint from {checkpoint_path}")
        detector.load_checkpoint(checkpoint_path)
        return detector

    def detector_releaser(detector):
        detector.to_device(torch.device("cpu"))
        torch.cuda.empty_cache()

    try:
        return collect_task_evidence(
            samples,
            tasks,
            preprocessors,
            detector_provider,
            batch_size,
            detector_releaser=detector_releaser,
        )
    finally:
        preprocessors.close()


def collect_checkpoint_evidence(
    samples,
    gpu_ids,
    tasks,
    detector_class,
    config,
    batch_size,
    checkpoint_root,
    log_root,
    seed,
    *,
    prepared_metadata=None,
):
    tasks_in_device = [
        task_group
        for task_group in round_robin_partition(tasks, len(gpu_ids))
        if task_group
    ]
    pending_results = []
    process_pool = mp.Pool(processes=len(tasks_in_device))
    try:
        for gpu_id, task_group in zip(gpu_ids, tasks_in_device):
            pending_results.append(
                process_pool.apply_async(
                    _collect_checkpoint_evidence_in_device,
                    (
                        gpu_id,
                        detector_class,
                        config,
                        samples,
                        task_group,
                        batch_size,
                        str(checkpoint_root),
                        log_root,
                        seed,
                    ),
                )
            )
        process_pool.close()
        process_pool.join()
    except Exception:
        process_pool.terminate()
        process_pool.join()
        raise

    image_paths = [sample.image.image_path for sample in samples]
    display_images = {
        path: metadata["display_image"]
        for path, metadata in (prepared_metadata or {}).items()
        if "display_image" in metadata
    }
    worker_results = []
    scoring_identities = []
    for pending_result in pending_results:
        worker_result, scoring_identity = pending_result.get()
        worker_results.append(worker_result)
        if scoring_identity is not None:
            scoring_identities.append(scoring_identity)
    if len(scoring_identities) != 1:
        raise ValueError("Exactly one dynamic detector scoring identity is required")
    evidence_by_path = merge_worker_evidence(
        worker_results,
        image_paths,
        display_images=display_images,
    )
    return evidence_by_path, scoring_identities[0]
