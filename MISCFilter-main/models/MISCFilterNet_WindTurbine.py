import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import *
import models.MISCKernel_cuda as misckernel


class LayerNorm2d(nn.Module):
    """PyTorch 1.8 兼容的 2D LayerNorm：在通道维做归一化。"""
    def __init__(self, num_channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # NCHW -> NHWC 做 LayerNorm，再转回 NCHW
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class LightVSSBlock(nn.Module):
    """
    轻量化 VSS 思想模块（纯 PyTorch 算子实现，兼容 1.8）：
    1) 先做 LayerNorm，提升状态更新稳定性；
    2) 用 1x1 Conv 做通道投影并门控分离（state / gate）；
    3) state 分支使用条带大核卷积 (1,9)+(9,1) 建模长程细长结构；
    4) 与 gate 分支逐点相乘实现选择性状态更新；
    5) 1x1 Conv 融合后残差回加。

    这种“门控 + 长程方向性扫描”的形式，等效模拟了 VSS/Mamba 中
    “选择性状态传播”的核心机制，尤其适合风机叶片这类细长旋转目标。
    """
    def __init__(self, channels):
        super(LightVSSBlock, self).__init__()
        self.norm = LayerNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, channels * 2, kernel_size=1, stride=1, padding=0, bias=True)

        # 深度可分离条带卷积：先横向再纵向，扩大感受野并保留方向先验
        self.strip_h = nn.Conv2d(channels, channels, kernel_size=(1, 9), stride=1, padding=(0, 4), groups=channels, bias=True)
        self.strip_v = nn.Conv2d(channels, channels, kernel_size=(9, 1), stride=1, padding=(4, 0), groups=channels, bias=True)

        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.norm(x)
        x = self.in_proj(x)

        x_state, x_gate = torch.chunk(x, 2, dim=1)

        x_state = self.act(x_state)
        x_state = self.strip_h(x_state)
        x_state = self.strip_v(x_state)

        # 门控分支使用 sigmoid，形成稳定的选择性写入
        x = x_state * torch.sigmoid(x_gate)
        x = self.out_proj(x)
        return x + identity


class KinematicMotionHead(nn.Module):
    """
    运动学旋转先验头：
    - 平移分支：预测全局线性平移 (tx, ty)
    - 旋转分支：预测角速度 omega 与旋转中心偏移 (cx_offset, cy_offset)

    前向中显式构建绝对坐标网格，并依据 v = w x r（二维一阶近似）生成局部旋转流场：
      vx = -omega * (y - cy)
      vy =  omega * (x - cx)

    最终输出 平移场 + 旋转场，作为动态核/偏移预测的物理先验引导。
    """
    def __init__(self, in_channels, hidden_channels=32):
        super(KinematicMotionHead, self).__init__()

        # 平移分支：卷积提取后用全局池化得到每张图像的全局平移参数
        self.translation_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, kernel_size=1, stride=1, padding=0, bias=True),
        )

        # 旋转分支：预测 omega, cx_offset, cy_offset
        self.rotation_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def _meshgrid(self, h, w, device, dtype):
        # 兼容旧版 torch.meshgrid API（PyTorch 1.8）
        ys = torch.arange(0, h, device=device, dtype=dtype)
        xs = torch.arange(0, w, device=device, dtype=dtype)
        if 'indexing' in torch.meshgrid.__code__.co_varnames:
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        else:
            grid_y, grid_x = torch.meshgrid(ys, xs)
        return grid_x, grid_y

    def forward(self, feat):
        b, _, h, w = feat.size()

        # 平移参数（每张图一个全局向量）
        trans = self.translation_branch(feat)
        trans = self.pool(trans)  # [B, 2, 1, 1]

        # 旋转参数（每张图一个全局旋转中心与角速度）
        rot = self.rotation_branch(feat)
        rot = self.pool(rot)      # [B, 3, 1, 1]
        omega = rot[:, 0:1, :, :]
        cx_offset = rot[:, 1:2, :, :]
        cy_offset = rot[:, 2:3, :, :]

        # 图像中心 + 偏移，得到旋转中心
        cx = (float(w - 1) * 0.5) + cx_offset
        cy = (float(h - 1) * 0.5) + cy_offset

        # 构建绝对坐标网格并计算相对半径向量 r
        grid_x, grid_y = self._meshgrid(h, w, feat.device, feat.dtype)
        grid_x = grid_x.view(1, 1, h, w)
        grid_y = grid_y.view(1, 1, h, w)

        rel_x = grid_x - cx
        rel_y = grid_y - cy

        # v = w x r 的二维一阶近似
        rot_vx = -omega * rel_y
        rot_vy = omega * rel_x
        rot_flow = torch.cat([rot_vx, rot_vy], dim=1)

        # 平移流场扩展到全分辨率后与旋转场融合
        trans_flow = trans.expand(b, 2, h, w)
        flow = trans_flow + rot_flow
        return flow


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8, ResBlock=ResBlock):
        super(EBlock, self).__init__()
        layers = [ResBlock(out_channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8, ResBlock=ResBlock):
        super(DBlock, self).__init__()
        layers = [ResBlock(channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel, BasicConv=BasicConv):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane, BasicConv=BasicConv, inchannel=3):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(inchannel, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane - inchannel, kernel_size=1, stride=1, relu=True)
        )
        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel, BasicConv=BasicConv):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


def CharbonnierFunc(data, epsilon=0.001):
    return torch.mean(torch.sqrt(data ** 2 + epsilon ** 2))


def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    device = flow.device

    if 'indexing' in torch.meshgrid.__code__.co_varnames:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=x.dtype),
            torch.arange(0, w, device=device, dtype=x.dtype),
            indexing='ij')
    else:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=x.dtype),
            torch.arange(0, w, device=device, dtype=x.dtype))
    grid = torch.stack((grid_x, grid_y), 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    grid_flow = grid_flow.type(x.type())
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output


class MISCKernelNet(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_blocks=[12, 12, 12],
                 num_blocks_kernel=[1, 1, 1],
                 kernel_size=7,
                 inference=False,
                 ):
        super(MISCKernelNet, self).__init__()
        self.inference = inference
        self.dim = dim
        self.kernel_size = kernel_size
        self.kernel_pad = int((self.kernel_size - 1) / 2.0)

        # 为了兼容原始工程中的 DOConv/推理路径，仅保留 BasicConv 的训练/推理切换。
        if not inference:
            BasicConv = BasicConv_do
        else:
            BasicConv = BasicConv_do_eval

        # 主干残差块统一替换为 LightVSSBlock（纯 PyTorch，避免 mamba_ssm 依赖）
        ResBlock = LightVSSBlock
        base_channel = dim

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_blocks[0], ResBlock=ResBlock),
            EBlock(base_channel * 2, num_blocks[1], ResBlock=ResBlock),
            EBlock(base_channel * 4, num_blocks[2], ResBlock=ResBlock),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(inp_channels, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4 * 2, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2 * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_blocks[2], ResBlock=ResBlock),
            DBlock(base_channel * 2, num_blocks[1], ResBlock=ResBlock),
            DBlock(base_channel, num_blocks[0], ResBlock=ResBlock)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel * 1, BasicConv=BasicConv),
            AFF(base_channel * 7, base_channel * 2, BasicConv=BasicConv)
        ])

        self.FAM1 = FAM(base_channel * 4, BasicConv=BasicConv)
        self.SCM1 = SCM(base_channel * 4, BasicConv=BasicConv)
        self.FAM2 = FAM(base_channel * 2, BasicConv=BasicConv)
        self.SCM2 = SCM(base_channel * 2, BasicConv=BasicConv)

        self.softmax = nn.Softmax(1)
        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])
        self.moduleKernel = misckernel.FunctionKernel.apply

        # 原始数据驱动 flow 预测头
        self.KernelPredictFlow = nn.ModuleList([
            BasicConv(base_channel * 4, 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel, 2, kernel_size=3, relu=False, stride=1),
        ])

        # 新增运动学先验头：在三尺度注入旋转先验流场
        self.KinematicHeads = nn.ModuleList([
            KinematicMotionHead(base_channel * 4, hidden_channels=max(16, base_channel // 2)),
            KinematicMotionHead(base_channel * 2, hidden_channels=max(16, base_channel // 2)),
            KinematicMotionHead(base_channel, hidden_channels=max(16, base_channel // 2)),
        ])

        self.flowup = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.KernelPredictFlowMask = nn.ModuleList([
            BasicConv(base_channel * 4, 1, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, 1, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel, 1, kernel_size=3, relu=False, stride=1),
        ])
        self.sigmoid = nn.Sigmoid()

        self.KernelOutBias = nn.ModuleList([
            BasicConv(base_channel * 4, out_channels, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, out_channels, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel, out_channels, kernel_size=3, relu=False, stride=1),
        ])

        self.KernelOutWeight = nn.ModuleList([
            BasicConv(base_channel * 4 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
        ])

        self.KernelOutkernelx = nn.ModuleList([
            BasicConv(base_channel * 4 * 2, kernel_size, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2 * 2, kernel_size, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, kernel_size, kernel_size=3, relu=False, stride=1),
        ])

        self.KernelOutkernely = nn.ModuleList([
            BasicConv(base_channel * 4 * 2, kernel_size, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2 * 2, kernel_size, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, kernel_size, kernel_size=3, relu=False, stride=1),
        ])

        self.KernelOutAlpha = nn.ModuleList([
            BasicConv(base_channel * 4 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
        ])

        self.KernelOutBeta = nn.ModuleList([
            BasicConv(base_channel * 4 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
        ])

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)

        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs_fil = list()
        outputs = list()
        Kernal_Loss = 0

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        z = self.Decoder[0](z)

        # ---------------- Scale 1/4：数据驱动 flow + 运动学旋转先验 ----------------
        s3_kernal_flow_data = self.KernelPredictFlow[0](z)
        s3_kernal_flow_prior = self.KinematicHeads[0](z)
        s3_kernal_flow = s3_kernal_flow_data + s3_kernal_flow_prior

        s3_kernal_flowmask = self.KernelPredictFlowMask[0](z)
        s3_kernal_flowmask = self.sigmoid(s3_kernal_flowmask)

        zx4 = torch.cat([z, x_4], 1)
        s3_kernal_flowfeat0, x_4_0 = torch.split(flow_warp(zx4, s3_kernal_flow.permute(0, 2, 3, 1)), self.dim * 4, dim=1)
        s3_kernal_flowfeat1, x_4_1 = torch.split(flow_warp(zx4, -s3_kernal_flow.permute(0, 2, 3, 1)), self.dim * 4, dim=1)
        x_4 = x_4_0 * s3_kernal_flowmask + x_4_1 * (1 - s3_kernal_flowmask)

        s3_kernal_bias = self.KernelOutBias[0](z)

        z = torch.cat([z, s3_kernal_flowfeat0 * s3_kernal_flowmask + s3_kernal_flowfeat1 * (1 - s3_kernal_flowmask)], 1)
        s3_kernal_weight = self.KernelOutWeight[0](z)
        s3_kernal_weight = self.softmax(s3_kernal_weight)
        s3_kernal_alpha = self.KernelOutAlpha[0](z)
        s3_kernal_beta = self.KernelOutBeta[0](z)
        s3_kernal_posx = self.KernelOutkernelx[0](z)
        s3_kernal_posy = self.KernelOutkernely[0](z)
        z = self.feat_extract[3](z)

        out3 = self.moduleKernel(self.modulePad(torch.cat([x_4, x_4.new_ones(x_4.size(0), 1, x_4.size(2), x_4.size(3))], 1)),
                                 s3_kernal_posx, s3_kernal_posy, s3_kernal_alpha, s3_kernal_beta, s3_kernal_weight)
        out3_norm = out3[:, -1:, :, :]
        out3_norm[out3_norm.abs() < 0.01] = 1.0
        out3 = out3[:, :-1, :, :] / out3_norm
        out3 += s3_kernal_bias
        if not self.inference:
            outputs.append(out3)
            outputs_fil.append(x_4)

            s3_Alpha = torch.mean(s3_kernal_weight * s3_kernal_alpha, dim=1, keepdim=True)
            s3_Beta = torch.mean(s3_kernal_weight * s3_kernal_beta, dim=1, keepdim=True)
            loss_s3_Alpha = CharbonnierFunc(s3_Alpha[:, :, :, :-1] - s3_Alpha[:, :, :, 1:]) + CharbonnierFunc(
                s3_Alpha[:, :, :-1, :] - s3_Alpha[:, :, 1:, :])
            loss_s3_Beta = CharbonnierFunc(s3_Beta[:, :, :, :-1] - s3_Beta[:, :, :, 1:]) + CharbonnierFunc(
                s3_Beta[:, :, :-1, :] - s3_Beta[:, :, 1:, :])
            Kernal_Loss += loss_s3_Alpha
            Kernal_Loss += loss_s3_Beta

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)

        # ---------------- Scale 1/2：融合上一级 flow + 当前运动学先验 ----------------
        s2_kernal_flow_data = self.KernelPredictFlow[1](z) + self.flowup(s3_kernal_flow) * 2
        s2_kernal_flow_prior = self.KinematicHeads[1](z)
        s2_kernal_flow = s2_kernal_flow_data + s2_kernal_flow_prior

        s2_kernal_flowmask = self.KernelPredictFlowMask[1](z)
        s2_kernal_flowmask = self.sigmoid(s2_kernal_flowmask)

        zx2 = torch.cat([z, x_2], 1)
        s2_kernal_flowfeat0, x_2_0 = torch.split(flow_warp(zx2, s2_kernal_flow.permute(0, 2, 3, 1)), self.dim * 2, dim=1)
        s2_kernal_flowfeat1, x_2_1 = torch.split(flow_warp(zx2, -s2_kernal_flow.permute(0, 2, 3, 1)), self.dim * 2, dim=1)
        x_2 = x_2_0 * s2_kernal_flowmask + x_2_1 * (1 - s2_kernal_flowmask)

        s2_kernal_bias = self.KernelOutBias[1](z)

        z = torch.cat([z, s2_kernal_flowfeat0 * s2_kernal_flowmask + s2_kernal_flowfeat1 * (1 - s2_kernal_flowmask)], 1)
        s2_kernal_weight = self.KernelOutWeight[1](z)
        s2_kernal_weight = self.softmax(s2_kernal_weight)
        s2_kernal_alpha = self.KernelOutAlpha[1](z)
        s2_kernal_beta = self.KernelOutBeta[1](z)
        s2_kernal_posx = self.KernelOutkernelx[1](z)
        s2_kernal_posy = self.KernelOutkernely[1](z)
        z = self.feat_extract[4](z)

        out2 = self.moduleKernel(self.modulePad(torch.cat([x_2, x_2.new_ones(x_2.size(0), 1, x_2.size(2), x_2.size(3))], 1)),
                                 s2_kernal_posx, s2_kernal_posy, s2_kernal_alpha, s2_kernal_beta, s2_kernal_weight)
        out2_norm = out2[:, -1:, :, :]
        out2_norm[out2_norm.abs() < 0.01] = 1.0
        out2 = out2[:, :-1, :, :] / out2_norm
        out2 += s2_kernal_bias
        if not self.inference:
            outputs.append(out2)
            outputs_fil.append(x_2)

            s2_Alpha = torch.mean(s2_kernal_weight * s2_kernal_alpha, dim=1, keepdim=True)
            s2_Beta = torch.mean(s2_kernal_weight * s2_kernal_beta, dim=1, keepdim=True)
            loss_s2_Alpha = CharbonnierFunc(s2_Alpha[:, :, :, :-1] - s2_Alpha[:, :, :, 1:]) + CharbonnierFunc(
                s2_Alpha[:, :, :-1, :] - s2_Alpha[:, :, 1:, :])
            loss_s2_Beta = CharbonnierFunc(s2_Beta[:, :, :, :-1] - s2_Beta[:, :, :, 1:]) + CharbonnierFunc(
                s2_Beta[:, :, :-1, :] - s2_Beta[:, :, 1:, :])
            Kernal_Loss += loss_s2_Alpha
            Kernal_Loss += loss_s2_Beta

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)

        z = self.Decoder[2](z)

        # ---------------- Full Scale：融合上一级 flow + 当前运动学先验 ----------------
        s1_kernal_flow_data = self.KernelPredictFlow[2](z) + self.flowup(s2_kernal_flow) * 2
        s1_kernal_flow_prior = self.KinematicHeads[2](z)
        s1_kernal_flow = s1_kernal_flow_data + s1_kernal_flow_prior

        s1_kernal_flowmask = self.KernelPredictFlowMask[2](z)
        s1_kernal_flowmask = self.sigmoid(s1_kernal_flowmask)

        zx = torch.cat([z, x], 1)
        s1_kernal_flowfeat0, x_0 = torch.split(flow_warp(zx, s1_kernal_flow.permute(0, 2, 3, 1)), self.dim, dim=1)
        s1_kernal_flowfeat1, x_1 = torch.split(flow_warp(zx, -s1_kernal_flow.permute(0, 2, 3, 1)), self.dim, dim=1)
        x = x_0 * s1_kernal_flowmask + x_1 * (1 - s1_kernal_flowmask)

        s1_kernal_bias = self.KernelOutBias[2](z)
        z = torch.cat([z, s1_kernal_flowfeat0 * s1_kernal_flowmask + s1_kernal_flowfeat1 * (1 - s1_kernal_flowmask)], 1)
        s1_kernal_weight = self.KernelOutWeight[2](z)
        s1_kernal_weight = self.softmax(s1_kernal_weight)
        s1_kernal_alpha = self.KernelOutAlpha[2](z)
        s1_kernal_beta = self.KernelOutBeta[2](z)
        s1_kernal_posx = self.KernelOutkernelx[2](z)
        s1_kernal_posy = self.KernelOutkernely[2](z)

        out = self.moduleKernel(self.modulePad(torch.cat([x, x.new_ones(x.size(0), 1, x.size(2), x.size(3))], 1)),
                                s1_kernal_posx, s1_kernal_posy, s1_kernal_alpha, s1_kernal_beta, s1_kernal_weight)
        out_norm = out[:, -1:, :, :]
        out_norm[out_norm.abs() < 0.01] = 1.0
        out = out[:, :-1, :, :] / out_norm
        out += s1_kernal_bias
        if not self.inference:
            outputs.append(out)
            outputs_fil.append(x)

            s1_Alpha = torch.mean(s1_kernal_weight * s1_kernal_alpha, dim=1, keepdim=True)
            s1_Beta = torch.mean(s1_kernal_weight * s1_kernal_beta, dim=1, keepdim=True)
            loss_s1_Alpha = CharbonnierFunc(s1_Alpha[:, :, :, :-1] - s1_Alpha[:, :, :, 1:]) + CharbonnierFunc(
                s1_Alpha[:, :, :-1, :] - s1_Alpha[:, :, 1:, :])
            loss_s1_Beta = CharbonnierFunc(s1_Beta[:, :, :, :-1] - s1_Beta[:, :, :, 1:]) + CharbonnierFunc(
                s1_Beta[:, :, :-1, :] - s1_Beta[:, :, 1:, :])
            Kernal_Loss += loss_s1_Alpha
            Kernal_Loss += loss_s1_Beta

            return outputs[::-1], outputs_fil[::-1]
        else:
            return out
