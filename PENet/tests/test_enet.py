import pathlib
import random
import sys
import unittest
from dataclasses import dataclass

import numpy as np
import paddle
import torch

seed = 0
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

paddle.seed(seed)


work_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(work_dir))

from PENet.penet.model import ENet  # noqa: E402
from PENet.tests.model import ENet as ENetRef  # noqa: E402


@dataclass
class ModelArgs:
    convolutional_layer_encoding: str
    network_model: str


class ENetTestCase(unittest.TestCase):
    def setUp(self):
        self.args = ModelArgs("xyz", "e")

    def test_forward(self):
        model = ENet(self.args)
        model.set_state_dict(paddle.load("./enet.pdparams"))
        model.eval()
        # batch_size = 1

        # Create sample input tensors
        # input_rgb = paddle.randn([batch_size, 3, 320, 1216])
        # input_d = paddle.randn([batch_size, 1, 320, 1216])
        # input_position = paddle.randn([batch_size, 2, 320, 1216])
        # input_K = paddle.randn([batch_size, 3, 3])
        inputs = np.load("./inputs.npz")
        # print(inputs)
        rgb_np = inputs["arr_5"]
        d_np = inputs["arr_4"]
        position_np = inputs["arr_1"]
        K_np = inputs["arr_0"]

        input_rgb = paddle.to_tensor(rgb_np)
        input_d = paddle.to_tensor(d_np)
        input_position = paddle.to_tensor(position_np)
        input_K = paddle.to_tensor(K_np)

        input_data = {
            "rgb": input_rgb,
            "d": input_d,
            "position": input_position,
            "K": input_K,
        }

        # Forward pass
        output = model(input_data)

        # Compare with reference implementation
        model_ref = ENetRef(self.args)
        model_ref.load_state_dict(torch.load("./e.pth.tar")["model"])
        model_ref.eval()
        input_rgb_ref = torch.from_numpy(rgb_np)
        input_d_ref = torch.from_numpy(d_np)
        input_position_ref = torch.from_numpy(position_np)
        input_K_ref = torch.from_numpy(K_np)
        input_data_ref = {
            "rgb": input_rgb_ref,
            "d": input_d_ref,
            "position": input_position_ref,
            "K": input_K_ref,
        }
        output_ref = model_ref(input_data_ref)

        # Compare outputs
        np.testing.assert_allclose(
            output[0].detach().numpy(),
            output_ref[0].detach().numpy(),
            atol=1e-5,
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
