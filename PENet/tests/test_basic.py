import paddle
import torch
import numpy as np

from basic_base import (
    BasicBlockBase,
    CSPNGenerateBase,
    CSPNGenerateAccelerateBase,
    BasicBlockGeoBase,
)
from basic_raw import (
    BasicBlockRaw,
    CSPNGenerateRaw,
    CSPNGenerateAccelerateRaw,
    BasicBlockGeoRaw,
)

from padiff import create_model, auto_diff

import unittest


def test_BasicBlock():
    module = create_model(BasicBlockBase(320, 320))
    module.auto_layer_map("base")
    layer = create_model(BasicBlockRaw(320, 320))
    layer.auto_layer_map("raw")

    input = np.random.randn(1, 320, 320, 3).astype("float32")
    inp = ({"x": torch.as_tensor(input)}, {"x": paddle.to_tensor(input)})
    assert (
        auto_diff(module, layer, inp, auto_weights=True, atol=1e-4) is True
    ), "Failed. expected success."


# def test_SparseDownSampleClose():
#     module = create_model(SparseDownSampleCloseBase(1))
#     module.auto_layer_map("base")
#     layer = create_model(SparseDownSampleCloseRaw(1))
#     layer.auto_layer_map("raw")


#     x = np.random.randn(1, 320, 320, 1).astype("float32")
#     y = np.random.randn(1, 320, 320, 1).astype("float32")
#     inp = ({"d": torch.as_tensor(x),
#             "mask": torch.as_tensor(y)},
#            {"d": paddle.to_tensor(x),
#             "mask": paddle.to_tensor(y)})
#     assert (
#         auto_diff(module, layer, inp, auto_weights=True, atol=1e-4) is True
#     ), "Failed. expected success."


def test_CSPNGenerate():
    module = create_model(CSPNGenerateBase(3, 5))
    module.auto_layer_map("base")
    layer = create_model(CSPNGenerateRaw(3, 5))
    layer.auto_layer_map("raw")

    x = np.random.randn(1, 3, 320, 320).astype("float32")
    inp = ({"feature": torch.as_tensor(x)}, {"feature": paddle.to_tensor(x)})
    assert (
        auto_diff(module, layer, inp, auto_weights=True, atol=1e-4) is True
    ), "Failed. expected success."


# def test_CSPN():
#     module = create_model(CSPNBase(5))
#     module.auto_layer_map("base")
#     layer = create_model(CSPNRaw(5))
#     layer.auto_layer_map("raw")

#     x = np.random.randn(1, 324, 324).astype("float32")
#     y = np.random.randn(1, 3, 320, 320).astype("float32")
#     z = np.random.randn(1, 3, 320, 320).astype("float32")

#     inp = (
#         {"guide_weight": torch.as_tensor(x),
#         "hn": torch.as_tensor(y),
#         "h0": torch.as_tensor(z)},
#         {"guide_weight": paddle.to_tensor(x),
#          "hn": paddle.to_tensor(y),
#          "h0": paddle.to_tensor(z)})
#     assert (
#         auto_diff(module, layer, inp, auto_weights=True, atol=1e-4) is True
#     ), "Failed. expected success."


def test_CSPNGenerateAccelerate():
    module = create_model(CSPNGenerateAccelerateBase(3, 5))
    module.auto_layer_map("base")
    layer = create_model(CSPNGenerateAccelerateRaw(3, 5))
    layer.auto_layer_map("raw")

    x = np.random.randn(1, 3, 320, 320).astype("float32")
    inp = ({"feature": torch.as_tensor(x)}, {"feature": paddle.to_tensor(x)})
    assert (
        auto_diff(module, layer, inp, auto_weights=True, atol=1e-4) is True
    ), "Failed. expected success."


# def test_CSPNAccelerate():
#     module = create_model(CSPNAccelerateBase(3))
#     module.auto_layer_map("base")
#     layer = create_model(CSPNAccelerateRaw(3))
#     layer.auto_layer_map("raw")

#     x = np.random.randn(1, 3, 3, 320, 320).astype("float32")
#     y = np.random.randn(1, 1, 320, 320).astype("float32")
#     z = np.random.randn(1, 320, 320).astype("float32")

#     inp = (
#         {"kernel": torch.as_tensor(x),
#         "input": torch.as_tensor(y),
#         "input0": torch.as_tensor(z)},
#         {"kernel": paddle.to_tensor(x),
#          "input": paddle.to_tensor(y),
#          "input0": paddle.to_tensor(z)})
#     assert (
#         auto_diff(module, layer, inp, auto_weights=True, atol=1e-4) is True
#     ), "Failed. expected success."


def test_BasicBlockGeo():
    module = create_model(BasicBlockGeoBase(320, 320))
    module.auto_layer_map("base")
    layer = create_model(BasicBlockGeoRaw(320, 320))
    layer.auto_layer_map("raw")

    x = np.random.randn(1, 320, 320, 3).astype("float32")
    g1 = np.random.randn(1, 3, 320, 3).astype("float32")
    g2 = np.random.randn(1, 3, 320, 3).astype("float32")

    inp = (
        {"x": torch.as_tensor(x), "g1": torch.as_tensor(g1), "g2": torch.as_tensor(g2)},
        {
            "x": paddle.to_tensor(x),
            "g1": paddle.to_tensor(g1),
            "g2": paddle.to_tensor(g2),
        },
    )
    assert (
        auto_diff(module, layer, inp, auto_weights=True, atol=1e-4) is True
    ), "Failed. expected success."


if __name__ == "__main__":
    unittest.main()
