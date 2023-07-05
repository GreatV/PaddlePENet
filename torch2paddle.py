from dataclasses import dataclass

import paddle
import torch

from PENet.penet.model import ENet


@dataclass
class ModelArgs:
    convolutional_layer_encoding: str
    network_model: str


def torch2paddle(torch_path, paddle_path):
    torch_ckpt = torch.load(torch_path, map_location="cpu")["model"]
    print(torch_ckpt.keys())

    args = ModelArgs("xyz", "e")
    model = ENet(args)
    model.eval()
    print(model.state_dict().keys())
    # paddle模型的字典keys中，act的weight和torch的权重不一样，需要调整
    # linear和torch的也不一致，需要转置
    layers_index = {}
    for name, param in model.named_sublayers():
        layers_index[name] = param

    paddle_ckpt = {}

    for k, v in torch_ckpt.items():
        pd_k = k
        pd_k = pd_k.replace("act.weight", "act._weight")
        pd_k = pd_k.replace("running_mean", "_mean")
        pd_k = pd_k.replace("running_var", "_variance")
        v = v.detach().cpu().numpy()

        if (
            isinstance(layers_index[pd_k.rsplit(".", 1)[0]], paddle.nn.Linear)
            and pd_k.rsplit(".", 1)[1] == "weight"
        ):
            new_shape = [1, 0] + list(range(2, v.ndim))
            print(
                "name: {}, ori shape: {}, new shape: {}".format(
                    k, v.shape, v.transpose(new_shape).shape
                )
            )
            v = v.transpose(new_shape)

        paddle_ckpt[pd_k] = v

    model.set_state_dict(paddle_ckpt)

    paddle.save(paddle_ckpt, paddle_path)


if __name__ == "__main__":
    # torch_path = "/home/greatx/repos/PaddlePENet/e.pth.tar"
    # paddle_path = r"enet.pdparams"
    torch_path = "/home/greatx/repos/PaddlePENet/pe.pth.tar"
    paddle_path = "penet.pdparams"
    torch2paddle(torch_path, paddle_path)
