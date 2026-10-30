"""Only the released final model's active layers, without numerical fixes."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class HCF(nn.Module):
    def __init__(self, in_features, out_features):
        super(HCF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 可学习参数
        self.C = nn.Parameter(torch.Tensor(out_features, in_features))
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.b = nn.Parameter(torch.Tensor(out_features))

        # p 参数初始化（提高数值稳定性）
        tsvalue = 2
        patch = (out_features - tsvalue) // 2
        remaining = out_features - tsvalue - 2 * patch
        
        # 使用较小的初始值以提高稳定性
        self.p0 = nn.Parameter(torch.empty(1, tsvalue).uniform_(0.2, 2.0))
        self.p1 = nn.Parameter(torch.empty(1, patch).uniform_(3.0, 5.0))
        self.p2 = nn.Parameter(torch.empty(1, out_features - tsvalue - patch).uniform_(2.0, 2.5))

        # s 参数（符号切换）
        quarter = out_features // 4
        s1 = torch.zeros(1, quarter)
        s2 = torch.ones(1, quarter)
        s3 = torch.zeros(1, quarter)
        s4 = torch.zeros(1, out_features - 3 * quarter)
        s = torch.cat([s1, s2, s3, s4], dim=1)
        self.register_buffer('s', s)

        # 更好的参数初始化
        nn.init.xavier_uniform_(self.C)
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

        # 激活函数
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        # x 的形状: [batch_size, node_n, out_channels, seq_len]
        # seq_len 是实际的特征维度（输入）
        
        # 保存原始形状，用于最后恢复
        original_shape = x.shape
        
        # 重塑为 [batch_size, in_features]
        bs, node_n, out_c, seq_len = original_shape
        # 合并批次维度
        x_flat = x.permute(0, 1, 2, 3).contiguous().view(-1, seq_len)

        # 运用层归一化
        x_flat = F.layer_norm(x_flat, [x_flat.size(-1)])
        
        # 扩展维度以支持广播: [batch_size, 1, in_features]
        x_expanded = x_flat.unsqueeze(1)
        
        # 拼接 p 参数
        p = torch.cat([self.p1, self.p0, self.p2], dim=1)  # [1, out_features]
        
        # 计算 q = (x - C) * W
        # [batch_size, 1, in_features] - [out_features, in_features] = [batch_size, out_features, in_features]
        q = (x_expanded - self.C.unsqueeze(0)) * self.W.unsqueeze(0)
        
        # 使用 tanh 作为符号函数的平滑近似（便于梯度传播）
        sign_q = torch.sign(q + 1e-8)     # 平滑的符号近似
        # print(f"sign_q after activation - min: {sign_q.min():.2f}, max: {sign_q.max():.2f}")
        
        # 对于 s=0，我们希望 sign^0 = 1；对于 s=1，我们希望 sign^1 = sign
        # 因此使用：result = s * sign + (1-s) * 1 = 1 + s * (sign - 1)
        s_expanded = self.s.unsqueeze(-1)  # [1, out_features, 1]
        sign_term = 1.0 + s_expanded * (sign_q - 1.0)
        
        # 计算绝对值项，并加入小量以提高数值稳定性
        abs_q = torch.abs(q) + 1e-4  # 加入小的 epsilon 保证稳定性
        
        p_expanded = p.unsqueeze(-1)  # [1, out_features, 1]
        
        # 使用对数-指数技巧实现更稳定的幂运算
        log_abs_q = torch.log(abs_q)
        abs_term = torch.exp(p_expanded * log_abs_q)
        # print(f"abs_term after activation - min: {abs_term.min():.2f}, max: {abs_term.max():.2f}")
        
        # 合并各项并求和
        combined = sign_term * abs_term
        q1 = combined.sum(dim=2)  # [batch_size, out_features]
        
        # 减去偏置项
        output = q1 - self.b.unsqueeze(0)
        
        # 应用最终激活函数以提高稳定性
        output = self.activation(output)
        # print(f"output after activation - min: {output.min():.2f}, max: {output.max():.2f}")
        
        # 恢复为原始形状
        output = output.contiguous().view(bs, node_n, out_c, self.out_features)
        
        return output

class QuaternionHCF(nn.Module):
    """
    四元数版本的HCF，用于处理通道维度
    使用四元数乘法而非普通矩阵乘法
    """
    def __init__(self, in_features, out_features):
        super(QuaternionHCF, self).__init__()
        assert in_features % 4 == 0, "in_features必须是4的倍数（四元数要求）"
        assert out_features % 4 == 0, "out_features必须是4的倍数（四元数要求）"
        
        self.in_features = in_features
        self.out_features = out_features
        self.quat_in = in_features // 4  # 四元数输入维度

        # 四元数权重参数（只需要1/4的权重）
        self.C = nn.Parameter(torch.Tensor(out_features, self.quat_in))
        self.W = nn.Parameter(torch.Tensor(out_features, self.quat_in))
        self.b = nn.Parameter(torch.Tensor(out_features))

        # p 参数初始化
        tsvalue = 2
        patch = (out_features - tsvalue) // 2
        remaining = out_features - tsvalue - 2 * patch
        
        self.p0 = nn.Parameter(torch.empty(1, tsvalue).uniform_(0.2, 2.0))
        self.p1 = nn.Parameter(torch.empty(1, patch).uniform_(3.0, 5.0))
        self.p2 = nn.Parameter(torch.empty(1, out_features - tsvalue - patch).uniform_(2.0, 2.5))

        # s 参数（符号切换）
        quarter = out_features // 4
        s1 = torch.zeros(1, quarter)
        s2 = torch.ones(1, quarter)
        s3 = torch.zeros(1, quarter)
        s4 = torch.zeros(1, out_features - 3 * quarter)
        s = torch.cat([s1, s2, s3, s4], dim=1)
        self.register_buffer('s', s)

        # 参数初始化
        nn.init.xavier_uniform_(self.C)
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

        # 激活函数
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        # x 的形状: [batch_size, node_n, seq_len, in_features]
        # in_features 是通道维度（需要处理的维度）
        
        original_shape = x.shape
        bs, node_n, seq_len, in_c = original_shape
        
        # 重塑为 [batch_size*node_n*seq_len, in_features]
        x_flat = x.contiguous().view(-1, in_c)
        
        # 层归一化
        x_flat = F.layer_norm(x_flat, [x_flat.size(-1)])
        
        # 通过四元数Hamilton积扩展权重矩阵
        # C 和 W 的原始大小是 [out_features, quat_in]
        # 扩展后变成 [out_features, in_features]
        C_expanded = self.expand_quaternion_weight(self.C)  # [out_features, in_features]
        W_expanded = self.expand_quaternion_weight(self.W)  # [out_features, in_features]
        
        # 扩展维度以支持广播: [batch, 1, in_features]
        x_expanded = x_flat.unsqueeze(1)  # [batch, 1, in_features]
        
        # 拼接 p 参数
        p = torch.cat([self.p1, self.p0, self.p2], dim=1)  # [1, out_features]
        
        # 计算 q = (x - C) * W，使用扩展后的权重
        # C_expanded: [out_features, in_features]
        # W_expanded: [out_features, in_features]
        q = (x_expanded - C_expanded.unsqueeze(0)) * W_expanded.unsqueeze(0)
        # q: [batch, out_features, in_features]
        
        # 符号函数
        sign_q = torch.sign(q + 1e-8)
        
        # 应用s参数
        s_expanded = self.s.unsqueeze(-1)  # [1, out_features, 1]
        sign_term = 1.0 + s_expanded * (sign_q - 1.0)
        
        # 计算绝对值项
        abs_q = torch.abs(q) + 1e-4
        p_expanded = p.unsqueeze(-1)  # [1, out_features, 1]
        
        # 稳定的幂运算
        log_abs_q = torch.log(abs_q)
        abs_term = torch.exp(p_expanded * log_abs_q)
        
        # 合并各项并求和
        combined = sign_term * abs_term
        q1 = combined.sum(dim=2)  # [batch, out_features]
        
        # 减去偏置项
        output = q1 - self.b.unsqueeze(0)
        
        # 应用激活函数
        output = self.activation(output)
        
        # 恢复为原始形状
        output = output.contiguous().view(bs, node_n, seq_len, self.out_features)
        
        return output
    
    def expand_quaternion_weight(self, weight):
        """
        通过四元数Hamilton积扩展权重矩阵
        输入: [out_features, quat_in]，其中 quat_in = in_features // 4
        输出: [out_features, in_features]
        
        使用Hamilton四元数乘法的标准结构来扩展权重
        """
        # weight: [out_features, quat_in]
        out_f, quat_in = weight.shape
        
        # 转置以便应用 make_quaternion_mul
        # weight_t: [quat_in, out_features]
        weight_t = weight.t()
        
        # 应用四元数Hamilton积
        # 输出: [in_features, out_features]
        hamilton = make_quaternion_mul(weight_t)
        
        # 转置回原来的维度顺序
        # expanded: [out_features, in_features]
        expanded = hamilton.t()
        
        return expanded

class GraphConvolution(nn.Module):
    """
    图卷积层，结合了空间图卷积和时间维度全连接
    输入形状: [batch_size, in_channels, num_nodes, seq_len]
    输出形状: [batch_size, out_channels, num_nodes, seq_len]
    """
    def __init__(self, in_c, out_c, node_n=22, node_n_s2=10, node_n_s3=5, seq_len=35, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_c
        self.out_features = out_c
        
        # 可学习的邻接矩阵参数
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        
        # 时间维度的全连接权重
        # self.weight_seq = Parameter(torch.FloatTensor(seq_len, seq_len))
        self.hcf = HCF(in_features=seq_len, out_features=seq_len)
        
        # 通道变换权重
        self.weight_c = Parameter(torch.FloatTensor(in_c, out_c))
        
        # 偏置项
        if bias:
            self.bias = Parameter(torch.FloatTensor(seq_len))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        """参数初始化"""
        stdv = 1. / math.sqrt(self.att.size(1))
        self.weight_c.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
        前向传播
        1. 根据输入节点数选择对应的邻接矩阵
        2. 执行图卷积操作
        3. 执行时间维度的全连接操作
        """
        # [bs, in_c, node_n, seq_len]
        support = torch.matmul(self.att, input.permute(0, 3, 2, 1)) # [bs, seq_len, node_n, in_c]
        output_gcn = torch.matmul(support, self.weight_c) # [bs, seq_len, node_n, out_c]
        # print(f"output_gcn after activation - min: {output_gcn.min():.2f}, max: {output_gcn.max():.2f}")
        output_fc = self.hcf(output_gcn.permute(0, 2, 3, 1)) # [bs, node_n, out_c, seq_len]
        output_fc = output_fc.permute(0, 2, 1, 3).contiguous() # [bs, out_c, node_n, seq_len]
        # print(f"output after activation - min: {output_fc.min():.2f}, max: {output_fc.max():.2f}")
        return output_fc
        
    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in_features} -> {self.out_features})'

class GCN_encoder(nn.Module):
    """
    GCN编码器
    包含初始图卷积层和多个残差块
    输入形状: [batch_size, in_channels, num_nodes, seq_len]
    输出形状: [batch_size, out_channels*2, num_nodes, seq_len]
    """
    def __init__(self, in_channal, out_channal, node_n=22, seq_len=25, p_dropout=0.3, num_stage=1, snum_stage=1):
        super(GCN_encoder, self).__init__()
        self.num_stage = num_stage
        self.snum_stage = snum_stage
        
        # 输入变换
        self.reshape_3_16 = torch.nn.Conv2d(in_channels=in_channal, out_channels=out_channal, kernel_size=(1, 1))
        
        # 批归一化层
        self.bn1 = nn.BatchNorm1d(out_channal * node_n * seq_len)    

        # 使用 Q4GNNLayer 进行输入维度的调整
        self.q4gnn1 = Q4GNNLayer(in_features=out_channal, out_features=out_channal, node_n=node_n, seq_len=seq_len, dropout=p_dropout)
        
        # 四元数图卷积残差块列表
        self.sqgcb = nn.ModuleList([
            QGC_Block(channal=out_channal, p_dropout=p_dropout, node_n=node_n, seq_len=seq_len)
            for _ in range(snum_stage)
        ])
        
        # 最后的四元数图卷积层，用于输出维度的调整
        self.q4gnn2 = Q4GNNLayer(in_features=out_channal, out_features=out_channal, node_n=node_n, seq_len=seq_len, dropout=p_dropout)
        
        # 输出通道变换
        self.reshape_conv = torch.nn.Conv2d(in_channels=in_channal, out_channels=out_channal, kernel_size=(1, 1))
        
        # 激活函数和Dropout
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        """前向传播，包含两个分支的融合"""
        # x : [bs, 3, self.node_n, dct_n]
        # # 主分支处理
        # y1 = self.gc1(x) # [bs, out_channal, self.node_n, dct_n]
        # b, c, n, l = y1.shape
        
        # y1 = y1.reshape(b, -1).contiguous() # [bs, out_channal * selc.node_n * dct_n]
        # y1 = self.bn1(y1).view(b, c, n, l).contiguous() # [bs, out_channal, self.node_n, dct_n]
        
        # y1 = self.act_f(y1)
        # y1 = self.do(y1)
        
        # # 通过残差块
        # for block in self.gcbs:
        #     y1 = block(y1)
        
        # y1 = self.gc7(y1)
        
        y = self.reshape_3_16(x) # [bs, out_channal, self.node_n, dct_n]
        y = self.q4gnn1(y) # [bs, out_channal, self.node_n, dct_n]
        b, c, n, l = y.shape

        y = y.reshape(b, -1).contiguous()
        y = self.bn1(y).view(b, c, n, l).contiguous()

        y = self.act_f(y)
        y = self.do(y)


        for block in self.sqgcb:
            y = block(y)

        y = self.q4gnn2(y)
        
        # 残差连接
        return y + self.reshape_conv(x)

class Q4GNNLayer(torch.nn.Module):
    """
    四元数图神经网络层
    输入形状: [batch_size, in_features, num_nodes, seq_len]
    输出形状: [batch_size, out_features, num_nodes, seq_len]
    """
    def __init__(self, in_features, out_features, dropout, node_n=22,
                  seq_len=35, quaternion_ff=True):
        super(Q4GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quaternion_ff = quaternion_ff
        self.dropout = nn.Dropout(dropout)
        
        # 邻接矩阵参数
        self.att = Parameter(torch.FloatTensor(node_n, node_n))

        # QuaternionHCF用于通道维度（使用四元数乘法）
        self.qhcf = QuaternionHCF(in_features=in_features, out_features=out_features)

        self.hcf = HCF(in_features=seq_len, out_features=seq_len)
        
        # 权重参数
        if self.quaternion_ff:
            self.weight = Parameter(torch.FloatTensor(self.in_features//4, self.out_features))
        else:
            self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))

        
        self.reset_parameters()

    def reset_parameters(self):
        """参数初始化"""
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)
        
        stdv_att = 1. / math.sqrt(self.att.size(1))
        self.att.data.uniform_(-stdv_att, stdv_att)

    def forward(self, input, double_type_used_in_graph=False):
        """前向传播，使用四元数乘法"""
        # [bs, in_c, node_n, seq_len]
        x = self.dropout(input).float()
        
        # 四元数乘法处理
        if self.quaternion_ff:
            hamilton = make_quaternion_mul(self.weight)
            if double_type_used_in_graph:
                hamilton = hamilton.double()
                x = x.double()
                adj = self.att.double()
            support = torch.matmul(x.permute(0, 3, 2, 1), hamilton) # [bs, seq_len, node_n, in_c]
        else:
            support = torch.mm(x, self.weight)
        
        output = torch.matmul(self.att, support) # [bs, seq_len, node_n, in_c]
        output = self.qhcf(output) # [bs, seq_len, node_n, in_c]
        output = output.permute(0, 3, 2, 1) # [bs, in_c, node_n, seq_len]
        output = self.hcf(output).contiguous()
        
        return output

class QGC_Block(nn.Module):
    """
    四元数图卷积残差块
    输入输出形状: [batch_size, channels, num_nodes, seq_len]
    """
    def __init__(self, channal, p_dropout, bias=True, node_n=22, node_n_s2=10, 
                 node_n_s3=5, seq_len=20):
        super(QGC_Block, self).__init__()
        self.in_features = channal
        self.out_features = channal
        
        # 四元数图卷积层
        self.q4gnn1 = Q4GNNLayer(channal, channal, node_n=node_n, seq_len=seq_len, dropout=p_dropout)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm1d(channal * node_n * seq_len)
        
        # 激活函数和Dropout
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        """前向传播，包含残差连接"""
        y = x
        y = self.q4gnn1(y)
        b, c, n, l = y.shape
        
        y = y.reshape(b, -1).contiguous()
        y = self.bn1(y).view(b, c, n, l).contiguous()
        
        y = self.act_f(y)
        y = self.do(y)
        
        return y + x

def make_quaternion_mul(kernel):
    """构造四元数乘法矩阵"""
    dim = kernel.size(1) // 4
    r, i, j, k = torch.split(kernel, [dim, dim, dim, dim], dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=0)
    i2 = torch.cat([i, r, -k, j], dim=0)
    j2 = torch.cat([j, k, r, -i], dim=0)
    k2 = torch.cat([k, -j, i, r], dim=0)
    hamilton = torch.cat([r2, i2, j2, k2], dim=1)
    return hamilton
