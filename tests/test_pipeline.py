import numpy as np
import torch

from train import CNN, grad_cam, near_duplicate_rate, scores


def test_model_output_shape():
    assert CNN()(torch.zeros(2, 1, 128, 128)).shape == (2, 4)


def test_scores_expose_imbalance():
    y = np.array([0] * 99 + [1])
    result = scores(y, np.zeros(100, dtype=int))
    assert result["accuracy"] == 0.99
    assert result["balanced_accuracy"] == 0.5


def test_near_duplicates_are_detected():
    rng = np.random.default_rng(0)
    train = rng.random((5, 1, 128, 128), dtype=np.float32)
    assert near_duplicate_rate(train, train[:2])["share_above"] == 1.0


def test_grad_cam_returns_one_map_per_image():
    x = np.random.default_rng(0).random((2, 1, 128, 128), dtype=np.float32)
    cam, pred = grad_cam(CNN(), x, "cpu")
    assert cam.shape == (2, 128, 128)
    assert pred.shape == (2,)
