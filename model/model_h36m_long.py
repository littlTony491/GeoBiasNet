"""Original released GeoBiasNet. The full-sequence GT reshape is intentional."""

from torch.nn import Module
from torch import nn
import torch
import utils.util as util
from siMLPe import mlp_4d
from model import BaseModel as BaseBlock

class MultiStageModel_4(Module):
    def __init__(self, opt):
        super(MultiStageModel_4, self).__init__()

        self.opt = opt
        self.kernel_size = opt.kernel_size
        self.d_model = opt.d_model
        self.dct_n = opt.dct_n
        self.in_features = opt.in_features
        self.num_stage = opt.num_stage
        self.node_n = self.in_features // 3
        self.input_n = opt.input_n
        self.output_n = opt.train_output_n

        # 模型参数验证
        assert opt.kernel_size == 10, "Kernel size must be 10"
        
        # 权重参数
        self.w1 = nn.Parameter(torch.tensor(0.07, requires_grad=True))
        self.w2 = nn.Parameter(torch.tensor(0.28, requires_grad=True))
        self.w3 = nn.Parameter(torch.tensor(0.65, requires_grad=True))
        self.w4 = nn.Parameter(torch.tensor(0.1, requires_grad=True))

        # 编码器-解码器结构
        self.encoder_layer_num = 1
        
        # 卷积层用于特征融合
        self.conv70_35 = torch.nn.Conv2d(in_channels=self.dct_n*2, out_channels=self.dct_n, kernel_size=(1,1))
        
        # 四个编码器分支
        enc_config = {
            'in_channal': 3, 
            'out_channal': self.d_model,
            'node_n': self.node_n,
            'seq_len': self.dct_n,
            'p_dropout': opt.drop_out,
            'num_stage': self.encoder_layer_num,
            'snum_stage': self.encoder_layer_num
        }
        self.gcn_encoder1 = BaseBlock.GCN_encoder(**enc_config)
        self.final_proj1 = nn.Conv2d(self.d_model, 3, kernel_size=1)
        self.gcn_encoder2 = BaseBlock.GCN_encoder(**enc_config)
        self.final_proj2 = nn.Conv2d(self.d_model, 3, kernel_size=1)
        self.gcn_encoder3 = BaseBlock.GCN_encoder(**enc_config)
        self.final_proj3 = nn.Conv2d(self.d_model, 3, kernel_size=1)
        self.gcn_encoder4 = BaseBlock.GCN_encoder(**enc_config)
        self.final_proj4 = nn.Conv2d(self.d_model, 3, kernel_size=1)

        # 窗口大小参数
        self.window_size = 6
        
        # 动态确定需要更新的帧数
        self.num_update_frames = self.input_n - (self.window_size - 2) + 1
        
        # 动态创建GCN残差模块
        self.gcn_blocks = nn.ModuleList([
            BaseBlock.GraphConvolution(
                in_c=3,
                out_c=3,
                node_n=self.node_n,
                seq_len=self.window_size - 2,
                bias=True
            ) for _ in range(self.num_update_frames)
        ])
        
        # MLP处理模块
        self.mlp_blockt = mlp_4d.MLPblock(
            dim=3, seq=self.window_size,
            use_norm=True, 
            use_spatial_fc=False,
            layernorm_axis='spatial'
        )
        self.mlp_blocks = mlp_4d.MLPblock(
            dim=3, seq=self.window_size,
            use_norm=True, 
            use_spatial_fc=True,
            layernorm_axis='temporal'
        )

        # 计算dct和idct需要用的矩阵
        dct_m, idct_m = util.get_dct_matrix(self.dct_n)
        dct_m = torch.from_numpy(dct_m).float()
        idct_m = torch.from_numpy(idct_m).float()
        self.register_buffer('dct_m', dct_m)
        self.register_buffer('idct_m', idct_m)

    def forward(self, src):
        bs = src.shape[0]
        dct_n = self.dct_n
        gt = src.clone()

        # 生成输入序列 [b,T,66]
        idx = list(range(self.kernel_size)) + [self.kernel_size - 1] * (dct_n - self.kernel_size)
        input_gcn1 = src[:, idx].clone()

        # 计算归一化权重
        total_weight = self.w1 + self.w2 + self.w3 + self.w4
        w1_normalized = self.w1 / total_weight
        w2_normalized = self.w2 / total_weight
        w3_normalized = self.w3 / total_weight
        w4_normalized = self.w4 / total_weight

        # 输入预处理 [b,66,dct_n]->[b,3,22,dct_n]
        input_gcn_dct = torch.einsum('ij,bjk->bik', self.dct_m[:dct_n], input_gcn1)
        input_gcn_dct = input_gcn_dct.permute(0, 2, 1)
        input_gcn_dct = input_gcn_dct.reshape(bs, self.node_n, -1, dct_n).permute(0, 2, 1, 3)

        # ================= 主处理分支（DCT域） =================
        # 第一级编码-解码
        latent_gcn_dct = self.gcn_encoder1(input_gcn_dct)
        output_dct_1 = self.final_proj1(latent_gcn_dct)[:, :, :, :dct_n]

        # 第二级编码-解码
        latent_gcn_dct = self.gcn_encoder2(output_dct_1)
        output_dct_2 = self.final_proj2(latent_gcn_dct)[:, :, :, :dct_n]

        # 第三级编码-解码
        latent_gcn_dct = self.gcn_encoder3(output_dct_2)
        output_dct_3 = self.final_proj3(latent_gcn_dct)[:, :, :, :dct_n]

        # 第四级编码-解码
        latent_gcn_dct = self.gcn_encoder4(output_dct_3)
        output_dct_4 = self.final_proj4(latent_gcn_dct)[:, :, :, :dct_n]

        # 加权融合四路输出（DCT域）
        com_out_dct = output_dct_1 * w1_normalized + output_dct_2 * w2_normalized + \
                      output_dct_3 * w3_normalized + output_dct_4 * w4_normalized  # [bs, 3, node_n, dct_n]

        # ================= 转换到时空域进行偏差更新 =================
        # 将融合后的DCT域结果转到时空域 [bs, 3, node_n, dct_n]
        com_out_dct_flat = com_out_dct.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)  # [bs, 66, dct_n]
        com_out_spatial = torch.matmul(self.idct_m[:, :dct_n], com_out_dct_flat.permute(0, 2, 1))  # [bs, dct_n, 66]
        com_out_spatial = com_out_spatial.permute(0, 2, 1).reshape(bs, self.node_n, -1, dct_n).permute(0, 2, 1, 3)  # [bs, 3, node_n, dct_n]
        com_out_spatial = torch.cat([com_out_spatial, com_out_spatial[:, :, :, -1:].clone()], dim=3) # [bs, 3, node_n, dct_n+1]
        
        # 准备gt的时空域版本
        gt_spatial = gt.reshape(bs, self.in_features // self.node_n, self.node_n, -1)
        
        # ================= 时空域残差更新（自适应 window_size 版本） =================
        output_updated = {}          # t最新版本 [B,3,N]
        update_count   = {}          # t已更新次数
        got_input2     = False       # 标志：self.input_n 是否已完成第二次更新

        C = self.window_size - 2     # 上下文帧数
        anchor_start = C                                 # 从第一个可以更新的位置开始
        anchor_end   = self.input_n + self.output_n - 1  # 让 a+1 能覆盖到输出末尾
        update_index = list(range(anchor_start, anchor_end + 1))

        for idx, a in enumerate(update_index):
            k_ctx_start = a - C      # 上下文起点（包含），长度为 C
            k_a         = a          # 目标 a
            k_ap1       = a + 1      # 目标 a+1

            use_residual_phase = not got_input2

            if use_residual_phase:
                # 组窗口：C 帧上下文 + a, a+1
                # 第一次用全原始；之后把 a 换成“更新过一次”的版本
                # 上下文：原始预测
                input_win = com_out_spatial[:, :, :, k_ctx_start:k_a]   # [B,3,N,C]

                if idx == 0:
                    # a 与 a+1 都用原始
                    tail = com_out_spatial[:, :, :, k_a:k_ap1+1]        # [B,3,N,2]
                else:
                    # a 用更新过一次的，a+1 仍用原始
                    tail_a   = output_updated[k_a].unsqueeze(-1)        # [B,3,N,1]
                    tail_ap1 = com_out_spatial[:, :, :, k_ap1].unsqueeze(-1)
                    tail = torch.cat([tail_a, tail_ap1], dim=-1)        # [B,3,N,2]

                input_win = torch.cat([input_win, tail], dim=-1)        # [B,3,N, C+2] == [B,3,N, self.window_size]

                # 残差 = 上下文“原始预测 - 原始GT”，仅对前 C 位注入
                ctx_pred = com_out_dct[:, :, :, k_ctx_start:k_a]        # [B,3,N,C]
                ctx_gt   = gt_spatial[:, :, :, k_ctx_start:k_a]         # [B,3,N,C]
                differ   = ctx_pred - ctx_gt                            # [B,3,N,C]

                out_dif  = self.gcn_blocks[idx](differ)                 # [B,3,N,C]
                mid      = self.mlp_blocks(input_win.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)                   # [B,3,N, self.window_size]
                mid[:, :, :, :C] += out_dif                             # 仅前 C 位注入残差
                out_seq  = self.mlp_blockt(mid.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)                         # [B,3,N, self.window_size]

                upd_a, upd_ap1 = out_seq[:, :, :, -2], out_seq[:, :, :, -1]  # 写回 a, a+1

            else:
                # 无残差阶段：C→2
                # 需要 C 帧“已更新两次”的上下文、1 帧“更新过一次”的 a、以及 1 帧“尚未更新”的 a+1
                g_ctx = [output_updated[t].unsqueeze(-1) for t in range(a - C, a)]  # C 个
                g_a   = output_updated[a].unsqueeze(-1)
                g_ap1 = com_out_spatial[:, :, :, k_ap1].unsqueeze(-1)
                input_win = torch.cat(g_ctx + [g_a, g_ap1], dim=-1)     # [B,3,N, self.window_size]

                mid     = self.mlp_blocks(input_win.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
                out_seq = self.mlp_blockt(mid.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
                upd_a, upd_ap1 = out_seq[:, :, :, -2], out_seq[:, :, :, -1]

            # 写回 a / a+1
            for t_idx, t_val in [(k_a, upd_a), (k_ap1, upd_ap1)]:
                c = update_count.get(t_idx, 0)
                output_updated[t_idx] = t_val
                update_count[t_idx]   = c + 1

            # 一旦 self.input_n 完成第二次更新，后续全走“无残差 C→2”
            if update_count.get(self.input_n, 0) >= 2:
                got_input2 = True

        # 收集最终区间 [input_n, input_n+output_n)
        outs = [output_updated[t].unsqueeze(-1) for t in range(self.input_n, self.input_n + self.output_n)]
        output_concatenated_spatial = torch.cat(outs, dim=-1)  # [B,3,N,output_n]

        # 与历史 input_n 帧拼接（其余管线保持不变）
        output_spatial_fin = torch.cat(
            [com_out_spatial[:, :, :, :self.input_n], output_concatenated_spatial], dim=-1
        )

        # ================= 输出处理（时空域） =================
        # 将时空域结果转换为输出格式 [bs, 3, node_n, dct_n] -> [bs, dct_n, 66]
        output_spatial_fin_flat = output_spatial_fin.permute(0, 3, 2, 1).reshape(bs, dct_n, -1)  # [bs, dct_n, 66]
        output_dct_4 = output_dct_4.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)
        output_dct_3 = output_dct_3.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)
        output_dct_2 = output_dct_2.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)
        output_dct_1 = output_dct_1.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)

        output_4 = torch.matmul(self.idct_m[:, :dct_n], output_dct_4.permute(0, 2, 1))
        output_3 = torch.matmul(self.idct_m[:, :dct_n], output_dct_3.permute(0, 2, 1))
        output_2 = torch.matmul(self.idct_m[:, :dct_n], output_dct_2.permute(0, 2, 1))
        output_1 = torch.matmul(self.idct_m[:, :dct_n], output_dct_1.permute(0, 2, 1))

        output_fin = torch.cat((output_spatial_fin_flat, input_gcn1), dim=1)
        output_4 = torch.cat((output_4, input_gcn1), dim=1)
        output_3 = torch.cat((output_3, input_gcn1), dim=1)
        output_2 = torch.cat((output_2, input_gcn1), dim=1)
        output_1 = torch.cat((output_1, input_gcn1), dim=1)
        
        # 降维处理
        output_fin = self.conv70_35(output_fin.reshape(bs, dct_n * 2, self.node_n, -1)).reshape(bs, dct_n, -1)
        output_4 = self.conv70_35(output_4.reshape(bs, dct_n * 2, self.node_n, -1)).reshape(bs, dct_n, -1)
        output_3 = self.conv70_35(output_3.reshape(bs, dct_n * 2, self.node_n, -1)).reshape(bs, dct_n, -1)
        output_2 = self.conv70_35(output_2.reshape(bs, dct_n * 2, self.node_n, -1)).reshape(bs, dct_n, -1)
        output_1 = self.conv70_35(output_1.reshape(bs, dct_n * 2, self.node_n, -1)).reshape(bs, dct_n, -1)

        return output_fin, output_4, output_3, output_2, output_1
