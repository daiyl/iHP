import torch
import torch.nn as nn
import os
import sys

# Optional dependency is imported only when xLSTM is selected.
try:
    from xlstm import (xLSTMBlockStack, xLSTMBlockStackConfig,
                       sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig)
except ImportError as exc:
    raise ImportError('The real xlstm package is required; no LSTM fallback is allowed.') from exc
from layers.Embed import DataEmbedding

class Model(nn.Module):
    """xLSTM时间序列预测模型"""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = getattr(configs, 'enc_in', 7)
        self.c_out = getattr(configs, 'c_out', 7)
        self.d_model = getattr(configs, 'd_model', 128)

        # 1. 使用局部变量存储参数
        # 从配置中获取参数或使用默认值
        xlstm_heads = getattr(configs, 'n_heads', 8)  # 直接使用n_heads
        xlstm_conv_kernel = getattr(configs, 'conv_kernel', 4)  # 使用通用参数
        xlstm_ff_factor = getattr(configs, 'ff_factor', 1.3)  # 使用通用参数
        xlstm_layers = getattr(configs, 'e_layers', 2)  # 直接使用e_layers
        activation = getattr(configs, 'activation', 'gelu')

        # 2. 嵌入层（按时间维度进行嵌入，保持序列长度为 L）
        self.embedding = DataEmbedding(
            getattr(configs, 'enc_in', 1), self.d_model,
            getattr(configs, 'embed', 'timeF'),
            getattr(configs, 'freq', 'h'),
            getattr(configs, 'dropout', 0.1)
        )

        # 3. 创建xLSTM配置 - 使用局部变量
        xlstm_config = xLSTMBlockStackConfig(
            mlstm_block=None,  # 不使用mLSTM
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend=getattr(configs, 'xlstm_backend', 'vanilla'),
                    num_heads=xlstm_heads,  # 使用局部变量
                    conv1d_kernel_size=xlstm_conv_kernel,  # 使用局部变量
                    bias_init="powerlaw_blockdependent"
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=xlstm_ff_factor,  # 使用局部变量
                    act_fn=activation
                )
            ),
            context_length=self.seq_len,
            num_blocks=xlstm_layers,  # 使用局部变量
            embedding_dim=self.d_model,
            slstm_at=list(range(xlstm_layers))  # 使用局部变量
        )

        # 4. xLSTM核心层
        self.xlstm_stack = xLSTMBlockStack(xlstm_config)

        # 5. 输出层
        self.output = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.c_out)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 1. 嵌入层（保持时间为序列维度）
        x = self.embedding(x_enc, x_mark_enc)  # [batch, seq_len, d_model]

        # 2. xLSTM处理
        x = self.xlstm_stack(x)  # [batch, seq_len, d_model]

        # 3. 只取预测部分
        x = x[:, -self.pred_len:, :]  # [batch, pred_len, d_model]

        # 4. 输出层
        x = self.output(x)  # [batch, pred_len, c_out]

        return x
