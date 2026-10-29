import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # 从 configs 对象获取所有必要参数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # 输入特征维度
        self.c_out = configs.c_out  # 输出特征维度

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=128,
            batch_first=True,
            num_layers=1
        )

        # 输出层
        self.projection = nn.Linear(128, self.c_out * self.pred_len)

        # SELU激活函数
        self.selu = nn.SELU()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        输入: x_enc [batch_size, seq_len, enc_in]
        输出: [batch_size, pred_len, c_out]
        """
        # LSTM处理
        lstm_out, _ = self.lstm(x_enc)

        # 应用SELU激活函数
        lstm_out = self.selu(lstm_out)

        # 只取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]

        # 投影到输出空间
        output = self.projection(last_output)

        # 重塑为 [batch_size, pred_len, c_out]
        output = output.view(-1, self.pred_len, self.c_out)

        return output