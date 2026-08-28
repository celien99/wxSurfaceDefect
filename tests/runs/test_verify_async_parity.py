import pytest

from runs.verify_async_parity import parse_args


def test_parse_args_rejects_non_positive_batch_size():
    with pytest.raises(SystemExit):
        parse_args(["--data-root", "d", "--batch-size", "0"])


def test_parse_args_accepts_minimal_invocation():
    args = parse_args(["--data-root", "d"])
    assert args.data_root == "d"
    assert args.manifest == "test_uni.jsonl"
    assert args.gpus == "0"
