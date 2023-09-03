from dataclasses import dataclass

import paddle
import torch
import numpy as np

from model_base import PENet_C2_Base
from model_raw import PENet_C2

from padiff import create_model, auto_diff

import unittest


@dataclass
class ModelArgs:
    convolutional_layer_encoding: str
    network_model: str
    dilation_rate: int = 2


args = ModelArgs("xyz", "e")


# def test_ENet():
#     module = create_model(ENetBase(args))
#     module.auto_layer_map("base")
#     layer = create_model(ENet(args))
#     layer.auto_layer_map("raw")

#     rgb = np.random.randn(1, 3, 320, 1216).astype("float32")
#     d = np.random.randn(1, 1, 320, 1216).astype("float32")
#     position = np.random.randn(1, 2, 320, 1216).astype("float32")
#     k = np.random.randn(1, 3, 3).astype("float32")

#     input_torch = {
#         "rgb": torch.as_tensor(rgb),
#         "d": torch.as_tensor(d),
#         "position": torch.as_tensor(position),
#         "K": torch.as_tensor(k),
#     }
#     input_paddle = {
#         "rgb": paddle.to_tensor(rgb),
#         "d": paddle.to_tensor(d),
#         "position": paddle.to_tensor(position),
#         "K": paddle.to_tensor(k),
#     }
#     inp = ({"input": input_torch}, {"input": input_paddle})

#     assert (
#         auto_diff(module, layer, inp, auto_weights=True, atol=1e-4) is True
#     ), "Failed. expected success."


# args = ModelArgs("xyz", "pe", 1)

# def test_PENet_C1():
#     module = create_model(PENet_C1_Base(args))
#     module.auto_layer_map("base")
#     layer = create_model(PENet_C1(args))
#     layer.auto_layer_map("raw")

#     rgb = np.random.randn(2, 3, 320, 320).astype("float32")
#     d = np.random.randn(2, 1, 320, 320).astype("float32")
#     position = np.random.randn(2, 2, 320, 320).astype("float32")
#     k = np.random.randn(2, 3, 3).astype("float32")

#     input_torch = {
#         "rgb": torch.as_tensor(rgb),
#         "d": torch.as_tensor(d),
#         "position": torch.as_tensor(position),
#         "K": torch.as_tensor(k),
#     }
#     input_paddle = {
#         "rgb": paddle.to_tensor(rgb),
#         "d": paddle.to_tensor(d),
#         "position": paddle.to_tensor(position),
#         "K": paddle.to_tensor(k),
#     }
#     inp = ({"input": input_torch}, {"input": input_paddle})

#     assert (
#         auto_diff(module, layer, inp, auto_weights=True, atol=1e-4) is True
#     ), "Failed. expected success."


args = ModelArgs("xyz", "pe", 2)


def test_PENet_C2():
    module = create_model(PENet_C2_Base(args))
    module.auto_layer_map("base")
    layer = create_model(PENet_C2(args))
    layer.auto_layer_map("raw")

    rgb = np.random.randn(1, 3, 160, 576).astype("float32")
    d = np.random.randn(1, 1, 160, 576).astype("float32")
    position = np.random.randn(1, 2, 160, 576).astype("float32")
    k = np.random.randn(1, 3, 3).astype("float32")

    input_torch = {
        "rgb": torch.as_tensor(rgb),
        "d": torch.as_tensor(d),
        "position": torch.as_tensor(position),
        "K": torch.as_tensor(k),
    }
    input_paddle = {
        "rgb": paddle.to_tensor(rgb),
        "d": paddle.to_tensor(d),
        "position": paddle.to_tensor(position),
        "K": paddle.to_tensor(k),
    }
    inp = ({"input": input_torch}, {"input": input_paddle})

    # net = PENet_C2_Base(args)
    # res = net(input_torch)
    # print(res)

    # net = PENet_C2(args)
    # res = net(input_paddle)
    # print(res)

    assert (
        auto_diff(module, layer, inp, auto_weights=True, atol=1e-4) is True
    ), "Failed. expected success."


if __name__ == "__main__":
    unittest.main()
