import math
import pathlib
import sys
import unittest

import paddle
import paddle.nn as nn
import torch

work_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(work_dir))

# from PENet.penet.basic import CSPN  # noqa: E402
# from PENet.penet.basic import CSPNAccelerate  # noqa: E402
from PENet.penet.basic import BasicBlock  # noqa: E402
from PENet.penet.basic import BasicBlockGeo  # noqa: E402
from PENet.penet.basic import CSPNGenerate  # noqa: E402
from PENet.penet.basic import CSPNGenerateAccelerate  # noqa: E402
from PENet.penet.basic import GeometryFeature  # noqa: E402
from PENet.penet.basic import SparseDownSampleClose  # noqa: E402
from PENet.penet.basic import convbn  # noqa: E402
from PENet.penet.basic import convbnrelu  # noqa: E402
from PENet.penet.basic import deconvbn  # noqa: E402
from PENet.penet.basic import deconvbnrelu  # noqa: E402
from PENet.penet.basic import get_pads  # noqa: E402
from PENet.penet.basic import weights_init  # noqa: E402


class TestGetPads(unittest.TestCase):
    def test_list_length(self):
        pads = get_pads(kernel_size=3)
        self.assertEqual(len(pads), 9)

    def test_padding(self):
        pads = get_pads(kernel_size=3)
        x = paddle.randn((1, 1, 5, 5))
        for i in range(9):
            padded_x = pads[i](x)
            self.assertEqual(padded_x.shape, [1, 1, 7, 7])


class TestWeightsInit(unittest.TestCase):
    def test_conv2d(self):
        # Test initialization of Conv2D layer
        layer = nn.Conv2D(16, 3, 3, padding=1)
        weights_init(layer)
        self.assertAlmostEqual(paddle.mean(layer.weight).numpy(), 0.0, delta=0.1)
        self.assertAlmostEqual(
            paddle.std(layer.weight).numpy(), math.sqrt(2.0 / (16 * 3 * 3)), delta=0.1
        )
        self.assertTrue(paddle.all(layer.bias == 0))

    def test_conv2dtranspose(self):
        # Test initialization of Conv2DTranspose layer
        layer = nn.Conv2DTranspose(16, 3, 3, padding=1)
        weights_init(layer)
        self.assertAlmostEqual(paddle.mean(layer.weight).numpy(), 0.0, delta=0.1)
        self.assertAlmostEqual(
            paddle.std(layer.weight).numpy(), math.sqrt(2.0 / (16 * 3 * 3)), delta=0.1
        )
        self.assertTrue(paddle.all(layer.bias == 0))

    def test_batchnorm2d(self):
        # Test initialization of BatchNorm2D layer
        layer = nn.BatchNorm2D(16)
        weights_init(layer)
        self.assertTrue(paddle.all(layer.weight == 1))
        self.assertTrue(paddle.all(layer.bias == 0))


def convbnrelu_ref(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True),
    )


class TestConvBNReLU(unittest.TestCase):
    def test_output_shape(self):
        # Test output shape of convbnrelu function
        in_channels = 3
        out_channels = 16
        kernel_size = 3
        stride = 1
        padding = 1
        input_shape = (1, in_channels, 32, 32)
        x = paddle.randn(input_shape)
        conv = convbnrelu(in_channels, out_channels, kernel_size, stride, padding)
        output = conv(x)
        expected_shape = [1, out_channels, 32, 32]
        self.assertEqual(output.shape, expected_shape)

    def test_output_mean(self):
        # Test output mean of convbnrelu function
        in_channels = 3
        out_channels = 16
        kernel_size = 3
        stride = 1
        padding = 1
        input_shape = (1, in_channels, 32, 32)
        x = paddle.randn(input_shape)
        conv = convbnrelu(in_channels, out_channels, kernel_size, stride, padding)
        output = conv(x)
        conv_ref = convbnrelu_ref(
            in_channels, out_channels, kernel_size, stride, padding
        )
        output_ref = conv_ref(torch.from_numpy(x.numpy()))
        self.assertAlmostEqual(
            paddle.mean(output).detach().numpy(),
            torch.mean(output_ref).detach().numpy(),
            delta=5e-3,
        )


def deconvbnrelu_ref(
    in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1
):
    return torch.nn.Sequential(
        torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=False,
        ),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True),
    )


class TestDeconvBNReLU(unittest.TestCase):
    def test_output_shape(self):
        # Test output shape of deconvbnrelu function
        in_channels = 3
        out_channels = 16
        kernel_size = 3
        stride = 1
        padding = 1
        output_padding = 0
        input_shape = (1, in_channels, 32, 32)
        x = paddle.randn(input_shape)
        deconv = deconvbnrelu(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        output = deconv(x)
        expected_shape = [1, out_channels, 32, 32]
        self.assertEqual(output.shape, expected_shape)

    def test_output_mean(self):
        # Test output mean of deconvbnrelu function
        in_channels = 3
        out_channels = 16
        kernel_size = 3
        stride = 1
        padding = 1
        output_padding = 0
        input_shape = (1, in_channels, 32, 32)
        x = paddle.randn(input_shape)
        deconv = deconvbnrelu(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        output = deconv(x)
        deconv_ref = deconvbnrelu_ref(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        output_ref = deconv_ref(torch.from_numpy(x.numpy()))
        self.assertAlmostEqual(
            paddle.mean(output).detach().numpy(),
            torch.mean(output_ref).detach().numpy(),
            delta=5e-3,
        )


def convbn_ref(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        torch.nn.BatchNorm2d(out_channels),
    )


class TestConvBN(unittest.TestCase):
    def test_output_shape(self):
        # Test output shape of convbn function
        in_channels = 3
        out_channels = 16
        kernel_size = 3
        stride = 1
        padding = 1
        input_shape = (1, in_channels, 32, 32)
        x = paddle.randn(input_shape)
        conv = convbn(in_channels, out_channels, kernel_size, stride, padding)
        output = conv(x)
        expected_shape = [1, out_channels, 32, 32]
        self.assertEqual(output.shape, expected_shape)

    def test_output_mean(self):
        # Test output mean of convbn function
        in_channels = 3
        out_channels = 16
        kernel_size = 3
        stride = 1
        padding = 1
        input_shape = (1, in_channels, 32, 32)
        x = paddle.randn(input_shape)
        conv = convbn(in_channels, out_channels, kernel_size, stride, padding)
        output = conv(x)

        conv_ref = convbn_ref(in_channels, out_channels, kernel_size, stride, padding)
        output_ref = conv_ref(torch.from_numpy(x.numpy()))
        self.assertAlmostEqual(
            paddle.mean(output).detach().numpy(),
            torch.mean(output_ref).detach().numpy(),
            delta=5e-3,
        )


def deconvbn_ref(
    in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0
):
    return torch.nn.Sequential(
        torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=False,
        ),
        torch.nn.BatchNorm2d(out_channels),
    )


class TestDeconvBN(unittest.TestCase):
    def test_output_shape(self):
        # Test output shape of deconvbn function
        in_channels = 3
        out_channels = 16
        kernel_size = 4
        stride = 2
        padding = 1
        output_padding = 0
        input_shape = (1, in_channels, 32, 32)
        x = paddle.randn(input_shape)
        deconv = deconvbn(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        output = deconv(x)
        expected_shape = [1, out_channels, 64, 64]
        self.assertEqual(output.shape, expected_shape)

    def test_output_mean(self):
        # Test output mean of deconvbn function
        in_channels = 3
        out_channels = 16
        kernel_size = 4
        stride = 2
        padding = 1
        output_padding = 0
        input_shape = (1, in_channels, 32, 32)
        x = paddle.randn(input_shape)
        deconv = deconvbn(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        output = deconv(x)
        deconv_ref = deconvbn_ref(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        output_ref = deconv_ref(torch.from_numpy(x.numpy()))
        self.assertAlmostEqual(
            paddle.mean(output).detach().numpy(),
            torch.mean(output_ref).detach().numpy(),
            delta=1e-3,
        )


class BasicBlockRef(torch.nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlockRef, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
            # norm_layer = encoding.torch.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv3x3_ref(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3_ref(planes, planes)
        self.bn2 = norm_layer(planes)
        if stride != 1 or inplanes != planes:
            downsample = torch.nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes),
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def conv3x3_ref(
    in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False, padding=1
):
    """3x3 convolution with padding"""
    if padding >= 1:
        padding = dilation
    return torch.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1, groups=1, bias=False):
    """1x1 convolution"""
    return torch.nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=bias
    )


class TestBasicBlock(unittest.TestCase):
    def test_output_shape(self):
        # Test output shape of BasicBlock class
        inplanes = 16
        planes = 16
        input_shape = (1, inplanes, 32, 32)
        x = paddle.randn(input_shape)
        block = BasicBlock(inplanes, planes)
        output = block(x)
        expected_shape = [1, planes, 32, 32]
        self.assertEqual(output.shape, expected_shape)

    def test_output_mean(self):
        # Test output mean of BasicBlock class
        inplanes = 16
        planes = 16
        input_shape = (1, inplanes, 32, 32)
        x = paddle.randn(input_shape)
        block = BasicBlock(inplanes, planes)
        output = block(x)
        block_ref = BasicBlockRef(inplanes, planes)
        output_ref = block_ref(torch.from_numpy(x.numpy()))
        self.assertAlmostEqual(
            paddle.mean(output).detach().numpy(),
            torch.mean(output_ref).detach().numpy(),
            delta=5e-2,
        )


class SparseDownSampleCloseRef(torch.nn.Module):
    def __init__(self, stride):
        super(SparseDownSampleCloseRef, self).__init__()
        self.pooling = torch.nn.MaxPool2d(stride, stride)
        self.large_number = 600

    def forward(self, d, mask):
        encode_d = -(1 - mask) * self.large_number - d

        d = -self.pooling(encode_d)
        mask_result = self.pooling(mask)
        d_result = d - (1 - mask_result) * self.large_number

        return d_result, mask_result


class TestSparseDownSampleClose(unittest.TestCase):
    def test_output_shape(self):
        # Test output shape of SparseDownSampleClose class
        stride = 2
        input_shape = (1, 16, 32, 32)
        d = paddle.randn(input_shape)
        mask = paddle.randn(input_shape)
        block = SparseDownSampleClose(stride)
        output, mask_result = block(d, mask)
        expected_shape = [1, 16, 16, 16]
        self.assertEqual(output.shape, expected_shape)
        self.assertEqual(mask_result.shape, expected_shape)

    def test_output_mean(self):
        # Test output mean of SparseDownSampleClose class
        stride = 2
        input_shape = (1, 16, 32, 32)
        d = paddle.randn(input_shape)
        mask = paddle.randn(input_shape)
        block = SparseDownSampleClose(stride)
        output, mask_result = block(d, mask)

        block_ref = SparseDownSampleCloseRef(stride)
        output_ref, mask_result_ref = block_ref(
            torch.from_numpy(d.numpy()), torch.from_numpy(mask.numpy())
        )
        self.assertAlmostEqual(
            paddle.mean(output).detach().numpy(),
            torch.mean(output_ref).detach().numpy(),
            delta=1e-2,
        )
        self.assertAlmostEqual(
            paddle.mean(mask_result).detach().numpy(),
            torch.mean(mask_result_ref).detach().numpy(),
            delta=1e-2,
        )


gks = 5
pad1 = [i for i in range(gks * gks)]
shift = torch.zeros(gks * gks, 4)
for i in range(gks):
    for j in range(gks):
        top = i
        bottom = gks - 1 - i
        left = j
        right = gks - 1 - j
        pad1[i * gks + j] = torch.nn.ZeroPad2d((left, right, top, bottom))


gks2 = 3
pad2 = [i for i in range(gks2 * gks2)]
shift = torch.zeros(gks2 * gks2, 4)
for i in range(gks2):
    for j in range(gks2):
        top = i
        bottom = gks2 - 1 - i
        left = j
        right = gks2 - 1 - j
        pad2[i * gks2 + j] = torch.nn.ZeroPad2d((left, right, top, bottom))


gks3 = 7  # guide kernel size
pad3 = [i for i in range(gks3 * gks3)]
shift = torch.zeros(gks3 * gks3, 4)
for i in range(gks3):
    for j in range(gks3):
        top = i
        bottom = gks3 - 1 - i
        left = j
        right = gks3 - 1 - j
        pad3[i * gks3 + j] = torch.nn.ZeroPad2d((left, right, top, bottom))


class CSPNGenerateRef(torch.nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(CSPNGenerateRef, self).__init__()
        self.kernel_size = kernel_size
        self.generate = convbn_ref(
            in_channels,
            self.kernel_size * self.kernel_size - 1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, feature):
        guide = self.generate(feature)

        # normalization
        guide_sum = torch.sum(guide.abs(), dim=1).unsqueeze(1)
        guide = torch.div(guide, guide_sum)
        guide_mid = (1 - torch.sum(guide, dim=1)).unsqueeze(1)

        # padding
        weight_pad = [i for i in range(self.kernel_size * self.kernel_size)]
        for t in range(self.kernel_size * self.kernel_size):
            zero_pad = 0
            if self.kernel_size == 3:
                zero_pad = pad2[t]
            elif self.kernel_size == 5:
                zero_pad = pad1[t]
            elif self.kernel_size == 7:
                zero_pad = pad3[t]
            if t < int((self.kernel_size * self.kernel_size - 1) / 2):
                weight_pad[t] = zero_pad(guide[:, t : t + 1, :, :])
            elif t > int((self.kernel_size * self.kernel_size - 1) / 2):
                weight_pad[t] = zero_pad(guide[:, t - 1 : t, :, :])
            else:
                weight_pad[t] = zero_pad(guide_mid)

        guide_weight = torch.cat(
            [weight_pad[t] for t in range(self.kernel_size * self.kernel_size)], dim=1
        )
        return guide_weight


class TestCSPNGenerate(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.in_channels = 3
        self.kernel_size = 3
        self.height = 16
        self.width = 16
        self.feature = paddle.randn(
            [self.batch_size, self.in_channels, self.height, self.width]
        )

    def test_output_shape(self):
        model = CSPNGenerate(self.in_channels, self.kernel_size)
        output = model(self.feature)
        expected_shape = [
            self.batch_size,
            self.kernel_size * self.kernel_size,
            self.height + 2,
            self.width + 2,
        ]
        self.assertEqual(output.shape, expected_shape)

    def test_output_sum(self):
        model = CSPNGenerate(self.in_channels, self.kernel_size)
        output = model(self.feature)
        model_ref = CSPNGenerateRef(self.in_channels, self.kernel_size)
        output_ref = model_ref(torch.from_numpy(self.feature.numpy()))
        self.assertAlmostEqual(
            paddle.sum(output).detach().numpy(),
            torch.sum(output_ref).detach().numpy(),
            delta=1e-2,
        )


class CSPNRef(torch.nn.Module):
    def __init__(self, kernel_size):
        super(CSPNRef, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, guide_weight, hn, h0):
        # CSPN
        half = int(0.5 * (self.kernel_size * self.kernel_size - 1))
        result_pad = [i for i in range(self.kernel_size * self.kernel_size)]
        for t in range(self.kernel_size * self.kernel_size):
            zero_pad = 0
            if self.kernel_size == 3:
                zero_pad = pad2[t]
            elif self.kernel_size == 5:
                zero_pad = pad1[t]
            elif self.kernel_size == 7:
                zero_pad = pad3[t]
            if t == half:
                result_pad[t] = zero_pad(h0)
            else:
                result_pad[t] = zero_pad(hn)
        guide_result = torch.cat(
            [result_pad[t] for t in range(self.kernel_size * self.kernel_size)], dim=1
        )
        guide_result = torch.sum((guide_weight.mul(guide_result)), dim=1)
        guide_result = guide_result[
            :,
            int((self.kernel_size - 1) / 2) : -int((self.kernel_size - 1) / 2),
            int((self.kernel_size - 1) / 2) : -int((self.kernel_size - 1) / 2),
        ]

        return guide_result.unsqueeze(dim=1)


# class TestCSPN(unittest.TestCase):
#     def setUp(self):
#         self.kernel_size = 3
#         self.cspn = CSPN(self.kernel_size)
#         self.guide_weight = paddle.ones([1, 9, 32, 32])
#         self.hn = paddle.ones([1, 9, 32, 32])
#         self.h0 = paddle.ones([1, 9, 32, 32])

#     def test_forward(self):
#         output = self.cspn.forward(self.guide_weight, self.hn, self.h0)
#         self.assertEqual(output.shape, (1, 9, 30, 30))
#         self.assertTrue(paddle.allclose(output, paddle.ones([1, 9, 30, 30])))


class CSPNGenerateAccelerateRef(torch.nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(CSPNGenerateAccelerateRef, self).__init__()
        self.kernel_size = kernel_size
        self.generate = convbn_ref(
            in_channels,
            self.kernel_size * self.kernel_size - 1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, feature):
        guide = self.generate(feature)

        # normalization in standard CSPN
        #'''
        guide_sum = torch.sum(guide.abs(), dim=1).unsqueeze(1)
        guide = torch.div(guide, guide_sum)
        guide_mid = (1 - torch.sum(guide, dim=1)).unsqueeze(1)
        #'''
        # weight_pad = [i for i in range(self.kernel_size * self.kernel_size)]

        half1, half2 = torch.chunk(guide, 2, dim=1)
        output = torch.cat((half1, guide_mid, half2), dim=1)
        return output


class TestCSPNGenerateAccelerate(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.in_channels = 3
        self.height = 32
        self.width = 32
        self.kernel_size = 5
        self.feature = paddle.randn(
            shape=[self.batch_size, self.in_channels, self.height, self.width]
        )

    def test_output_shape(self):
        model = CSPNGenerateAccelerate(self.in_channels, self.kernel_size)
        output = model(self.feature)
        self.assertEqual(
            output.shape,
            [
                self.batch_size,
                self.kernel_size * self.kernel_size,
                self.height,
                self.width,
            ],
        )

    def test_output_mean(self):
        model = CSPNGenerateAccelerate(self.in_channels, self.kernel_size)
        output = model(self.feature)
        model_ref = CSPNGenerateAccelerateRef(self.in_channels, self.kernel_size)
        output_ref = model_ref(torch.from_numpy(self.feature.numpy()))

        self.assertAlmostEqual(
            paddle.mean(output).detach().numpy(),
            torch.mean(output_ref).detach().numpy(),
            delta=1e-2,
        )


class CSPNAccelerateRef(torch.nn.Module):
    def __init__(self, kernel_size, dilation=1, padding=1, stride=1):
        super(CSPNAccelerateRef, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(
        self, kernel, input, input0
    ):  # with standard CSPN, an addition input0 port is added
        bs = input.size()[0]
        h, w = input.size()[2], input.size()[3]
        input_im2col = torch.nn.F.unfold(
            input, self.kernel_size, self.dilation, self.padding, self.stride
        )
        kernel = kernel.reshape(bs, self.kernel_size * self.kernel_size, h * w)

        # standard CSPN
        input0 = input0.view(bs, 1, h * w)
        mid_index = int((self.kernel_size * self.kernel_size - 1) / 2)
        input_im2col[:, mid_index : mid_index + 1, :] = input0

        # print(input_im2col.size(), kernel.size())
        output = torch.einsum("ijk,ijk->ik", (input_im2col, kernel))
        return output.view(bs, 1, h, w)


# class TestCSPNAccelerate(unittest.TestCase):
#     def setUp(self):
#         self.batch_size = 2
#         self.in_channels = 3
#         self.height = 32
#         self.width = 32
#         self.kernel_size = 5
#         self.dilation = 1
#         self.padding = 1
#         self.stride = 1
#         self.kernel = paddle.randn(
#             shape=[
#                 self.batch_size,
#                 self.kernel_size * self.kernel_size,
#                 self.height * self.width,
#             ]
#         )
#         self.input = paddle.randn(
#             shape=[self.batch_size, self.in_channels, self.height, self.width]
#         )
#         self.input0 = paddle.randn(shape=[self.batch_size, self.height, self.width])

#     # def test_output_shape(self):
#     #     model = CSPNAccelerate(
#     #         self.kernel_size, self.dilation, self.padding, self.stride
#     #     )
#     #     output = model(self.kernel, self.input, self.input0)
#     #     self.assertEqual(output.shape, [self.batch_size, 1,
#       self.height, self.width])

#     def test_output_sum(self):
#         model = CSPNAccelerate(
#             self.kernel_size, self.dilation, self.padding, self.stride
#         )
#         output = model(self.kernel, self.input, self.input0)
#         model_ref = CSPNAccelerateRef(
#             self.kernel_size, self.dilation, self.padding, self.stride
#         )
#         output_ref = model_ref(
#             torch.from_numpy(self.kernel.numpy()),
#             torch.from_numpy(self.input.numpy()),
#             torch.from_numpy(self.input0.numpy()),
#         )
#         self.assertAlmostEqual(
#             paddle.mean(output).detach().numpy(),
#             torch.mean(output_ref).detach().numpy(),
#             delta=1e-2,
#         )


class GeometryFeatureRef(torch.nn.Module):
    def __init__(self):
        super(GeometryFeatureRef, self).__init__()

    def forward(self, z, vnorm, unorm, h, w, ch, cw, fh, fw):
        x = z * (0.5 * h * (vnorm + 1) - ch) / fh
        y = z * (0.5 * w * (unorm + 1) - cw) / fw
        return torch.cat((x, y, z), 1)


class TestGeometryFeature(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.z = paddle.randn(shape=[self.batch_size, 1, 32, 32])
        self.vnorm = paddle.randn(shape=[self.batch_size, 1, 32, 32])
        self.unorm = paddle.randn(shape=[self.batch_size, 1, 32, 32])
        self.h = 32
        self.w = 32
        self.ch = 16
        self.cw = 16
        self.fh = 32
        self.fw = 32

    def test_output_shape(self):
        model = GeometryFeature()
        output = model(
            self.z,
            self.vnorm,
            self.unorm,
            self.h,
            self.w,
            self.ch,
            self.cw,
            self.fh,
            self.fw,
        )
        self.assertEqual(output.shape, [self.batch_size, 3, self.h, self.w])

    def test_output_value(self):
        model = GeometryFeature()
        output = model(
            self.z,
            self.vnorm,
            self.unorm,
            self.h,
            self.w,
            self.ch,
            self.cw,
            self.fh,
            self.fw,
        )
        x_expected = self.z * (0.5 * self.h * (self.vnorm + 1) - self.ch) / self.fh
        y_expected = self.z * (0.5 * self.w * (self.unorm + 1) - self.cw) / self.fw
        output_expected = paddle.concat([x_expected, y_expected, self.z], axis=1)
        self.assertTrue(paddle.allclose(output, output_expected))


class BasicBlockGeoRef(torch.nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        geoplanes=3,
    ):
        super(BasicBlockGeoRef, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            # norm_layer = encoding.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv3x3_ref(inplanes + geoplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_ref(planes + geoplanes, planes)
        self.bn2 = norm_layer(planes)
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes + geoplanes, planes, stride),
                norm_layer(planes),
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, g1=None, g2=None):
        identity = x
        if g1 is not None:
            x = torch.cat((x, g1), 1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if g2 is not None:
            out = torch.cat((g2, out), 1)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class TestBasicBlockGeo(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.inplanes = 3
        self.planes = 6
        self.stride = 1
        self.groups = 1
        self.base_width = 64
        self.dilation = 1
        self.geoplanes = 3
        self.input = paddle.randn(shape=[self.batch_size, self.inplanes, 32, 32])
        self.g1 = paddle.randn(shape=[self.batch_size, self.geoplanes, 32, 32])
        self.g2 = paddle.randn(shape=[self.batch_size, self.geoplanes, 32, 32])

    def test_output_shape(self):
        model = BasicBlockGeo(
            self.inplanes,
            self.planes,
            self.stride,
            None,
            self.groups,
            self.base_width,
            self.dilation,
            None,
            self.geoplanes,
        )
        output = model(self.input, self.g1, self.g2)
        self.assertEqual(output.shape, [self.batch_size, self.planes, 32, 32])

    def test_output_value(self):
        model = BasicBlockGeo(
            self.inplanes,
            self.planes,
            self.stride,
            None,
            self.groups,
            self.base_width,
            self.dilation,
            None,
            self.geoplanes,
        )
        output = model(self.input, self.g1, self.g2)
        identity = self.input
        if self.g1 is not None:
            identity = paddle.concat((identity, self.g1), axis=1)
        out = paddle.nn.functional.conv2d(
            identity, model.conv1.weight, stride=self.stride, padding=1
        )
        out = model.bn1(out)
        out = paddle.nn.functional.relu(out)
        if self.g2 is not None:
            out = paddle.concat((self.g2, out), axis=1)
        out = paddle.nn.functional.conv2d(out, model.conv2.weight, stride=1, padding=1)
        out = model.bn2(out)
        if model.downsample is not None:
            identity = model.downsample(identity)
        out += identity
        out = paddle.nn.functional.relu(out)
        self.assertTrue(paddle.allclose(output, out))


if __name__ == "__main__":
    unittest.main()
