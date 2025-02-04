import math
import torch
from torch import nn
import torch.nn.functional as F

def FFT_for_Period(x, k=2):
    # [B, C, T]
    xf = torch.fft.rfft(x, dim=2)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(0)  
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()  
    period = x.shape[2] 
    return period, abs(xf).mean(1)[:, top_list]

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class PAFConvBlock(nn.Module):
    def __init__(self, d_model, d_base, kernel_list):
        super(PAFConvBlock, self).__init__()
        self.pafconv_list = nn.ModuleList()

        for kernel_size in kernel_list:
            pafconv = nn.Sequential(
                nn.Conv2d(1, d_base, kernel_size=3, padding='same', bias=False),
                nn.BatchNorm2d(d_base),
                nn.LeakyReLU(),
                # nn.LeakyReLU(),

                nn.Conv2d(d_base, d_model, kernel_size=kernel_size, padding='same', groups=d_base, bias=False),
                nn.BatchNorm2d(d_model),
                nn.LeakyReLU(),

                nn.Conv2d(d_model, d_base, kernel_size=3, padding='same', bias=False),
                nn.BatchNorm2d(d_base),
                nn.LeakyReLU()

                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            self.pafconv_list.append(pafconv)

    def forward(self, x, kernel_index):
        return self.pafconv_list[kernel_index](x)

class RegionConvBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding='valid', bias=False):
        super(RegionConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, in_channels * 4, kernel_size=kernel_size, groups=in_channels, padding=padding, stride=stride, bias=bias)
        self.bn1 = nn.BatchNorm1d(in_channels * 4)
        self.act = nn.LeakyReLU()

        self.dwconv = nn.Conv1d(in_channels * 4, in_channels, kernel_size=1, padding='same', bias=False)
        self.bn2 = nn.BatchNorm1d(in_channels)
    def forward(self, x):
        x = self.act(self.bn1(self.conv(x)))
        x = self.act(self.bn2(self.dwconv(x)))
        return x

class ChannelAttention(nn.Module):
    def __init__(self, d_model, reduction):
        super(ChannelAttention, self).__init__()
        self.pwconv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.Tanh())

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(d_model // reduction, d_model, bias=False))
    def forward(self, x):
        # x: [B, C, T]
        batch_size, d_model, _ = x.size()
        # Global average pooling and max pooling
        x = self.pwconv(x)
        avg_pool = F.adaptive_avg_pool1d(x, 1).view(batch_size, d_model)
        max_pool = F.adaptive_max_pool1d(x, 1).view(batch_size, d_model)

        # Channel attention
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)

        channel_attention = torch.sigmoid(avg_out + max_out).unsqueeze(2)
        # out = x * channel_attention
        out = torch.matmul(channel_attention.transpose(1, 2), x)   # [B, 1, T]
        return out


class TokenEmbedding(nn.Module):
    def __init__(self, config):
        super(TokenEmbedding, self).__init__()
        self.k = config['top_k']
        self.patch_pattern = config['patch_pattern']
        self.period_list = config['period']
        self.d_base = config['d_base']
        d_model = self.d_base * (len(self.period_list)+1)
        # self.channel_size, self.seq_len = config['Data_shape'][1], config['Data_shape'][2]
        seq_len = config['Data_shape'][2]
        self.k_list = config['kernel']
        self.pafconv = PAFConvBlock(d_model, self.d_base, self.k_list)

        self.fineconv = nn.Sequential(
            nn.Conv2d(1, self.d_base * 16, kernel_size=(1, 3), padding='same', bias=False),
            nn.BatchNorm2d(self.d_base * 16),
            nn.LeakyReLU(),  # nn.GELU(),

            nn.Conv2d(self.d_base * 16, self.d_base, kernel_size=(1, 8), padding='same', bias=False),
            nn.BatchNorm2d(self.d_base),
            nn.LeakyReLU()
        )

        self.use_pe = config['use_pe']
        self.position_embedding = PositionalEmbedding(d_model)
        self.reduconv = RegionConvBlock(d_model, kernel_size=seq_len, padding='valid', bias=False)
        # self.reduconv = nn.Sequential(
        #     nn.Conv1d(d_model, 1, kernel_size=4, padding='same', bias=False),
        #     nn.BatchNorm1d(1),
        #     nn.GELU(),
        #     nn.AdaptiveAvgPool1d(d_model))
        # self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # self.adpool = nn.AdaptiveAvgPool1d(d_model)
        # self.simconv = nn.Sequential(
        #     nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, padding='same', bias=False),
        #     nn.BatchNorm1d(d_model),
        #     nn.LeakyReLU(),
        #
        #     nn.Conv1d(d_model, 1, kernel_size=1, padding='same', bias=False),
        #     nn.BatchNorm1d(1),
        #     nn.LeakyReLU()
        # )
        # self.chan_att = ChannelAttention(d_model, reduction=4)

    def forward(self, x):
        batch_size = x.size(0)
        channel_size = x.size(1)
        seq_len = x.size(2)
        if self.patch_pattern == 'Fixed':
            period_list = self.period_list
        else:
            period_list, period_weight = FFT_for_Period(x, self.k)  # top list => frequency     period为patch长度

        paf = []
        for i in range(len(period_list)):
            period = period_list[i]    # patch_len
            # padding
            if seq_len % period != 0:
                length = ((seq_len // period) + 1) * period
                padding = torch.zeros([batch_size, channel_size, (length - seq_len)]).to(x.device)
                out = torch.cat([x, padding], dim=2)
            else:
                length = seq_len
                out = x

            # reshape
            out = out.view(-1, length).contiguous()
            out = out.view(-1, length // period, period).unsqueeze(1).contiguous()    # to make the tensor contiguous in memory, call contiguous()
            # => input = batch*channel * in_embed=1 * f * period (patch_len
            out = self.pafconv(out, i)
            # => output = batch*channel * out_embed= 4 * f * period (patch_len
            out = out.view(out.shape[0], out.shape[1], -1)
            paf.append(out[:, :, :seq_len])       # delete padding
            # => batch*channel * out_embed * seq_len
        finef = x.unsqueeze(1)
        # => input = batch * in_embed=1 * channel * seq_len
        finef = self.fineconv(finef).permute(0, 2, 1, 3).contiguous()
        # => output = batch * channel * out_embed= 4 * seq_len
        finef = finef.view(-1, finef.shape[2], seq_len).contiguous()
        paf.append(finef)
        # paf.append(x.view(-1, seq_len).unsqueeze(1))
        paf = torch.cat(paf, dim=1)
        # => batch_size*channel_size  d_model  seq_len
        # msf = self.chan_att(paf)
        # msf = self.simconv(paf)
        msf = self.reduconv(paf)
        # => batch_size*channel_size  1  seq_len
        # msf = self.adpool(msf)     # => batch_size*channel_size  1  d_model
        msf = msf.view(batch_size, channel_size, -1).contiguous()
        # => batch_size  channel_size  d_model
        msf = msf + self.position_embedding(msf) if self.use_pe else msf
        return msf


class CrossAttenBlock(nn.Module):
    def __init__(self, config, d_model, num_heads, dropout=0.1):
        super(CrossAttenBlock, self).__init__()
        self.num_heads = num_heads
        self.scale = d_model ** -0.5
        use_prenorm = config['use_prenorm']
        self.use_res = config['use_res']
        self.dataset = config['dataset']

        self.graph_biosemi = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7',
                              'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5','T7', 'TP7',
                              'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
                              'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
                              'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4',
                              'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
                              'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
                              'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
        self.graph_kd = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7',
                         'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
                         'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
                         'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'O2', 'PO8', 'PO4',
                         'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10',
                         'FC6', 'FC4', 'FC2', 'C2', 'C4', 'C6', 'T8', 'TP8',
                         'Fp2', 'AF8', 'AF4', 'F2', 'F4', 'F6', 'F8', 'FT8',
                         'Fpz', 'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz']


        self.graph_standard = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                               'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
                               'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2',
                               'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz',
                               'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4',
                               'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6',
                               'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8',
                               'TP7', 'TP8', 'PO7', 'PO8', 'FT9', 'FT10', 'Fpz', 'CPz', 'FCz']
        self.graph_asa = ['Fp1', 'AF7', 'AF3', 'F1', 'FC1', 'F3', 'F5', 'F7',
                          'FC3', 'FC5', 'FT7', 'FT9', 'C1', 'C3', 'C5', 'T7',
                          'TP7', 'CP5', 'CP3', 'CP1', 'TP9', 'P7', 'PO7', 'O1',
                          'P1', 'P3', 'P5', 'PO3', 'PO4', 'P6', 'P4', 'P2',
                          'TP8', 'CP6', 'CP4', 'CP2', 'TP10', 'P8', 'PO8', 'O2',
                          'FC4', 'FC6', 'FT8', 'FT10', 'C2', 'C4', 'C6', 'T8',
                          'Fp2', 'AF8', 'AF4', 'F2', 'FC2', 'F4', 'F6', 'F8',
                          'Fpz', 'Fz', 'Cz', 'Fz', 'CPz', 'Pz', 'POz', 'Oz']

        k_list = config['kernel_size']
        self.conv_region = nn.ModuleList()
        for k in k_list:
            conv = RegionConvBlock(d_model, kernel_size=k, stride=4, padding='valid', bias=False)
            self.conv_region.append(conv)
        # input: B d_model C  output: # B d_model C/8

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv1 = nn.Linear(d_model, d_model, bias=False)
        self.kv2 = nn.Linear(d_model, d_model, bias=False)

        self.start = nn.LayerNorm(d_model) if use_prenorm else nn.Identity()
        self.post = nn.Identity() if use_prenorm else nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def reorder_channel(self, data):
        """
        @copyright form LGGNet
           This function reorder the channel according to different graph designs
           Parameters
           ----------
           data: (batch_size, channel, data)
           graph: graph type

           Returns
           -------
           reordered data: (batch_size, channel, data)
        """
        if self.dataset == 'ASA':
            graph_idx = self.graph_asa
        else:
            graph_idx = self.graph_kd

        idx = []
        for chan in graph_idx:
            if self.dataset == 'ASA':
                idx.append(self.graph_standard.index(chan))
            else:
                idx.append(self.graph_biosemi.index(chan))
        return data[:, idx, :]


    def forward(self, x):
        B, C, D = x.shape
        if self.dataset == 'ASA':
            padding = torch.zeros([B, 1, D]).to(x.device)
            x = torch.cat([x, padding], dim=1)
            C = C+1

        x_n = self.start(x)

        x_n = self.reorder_channel(x_n)

        q = self.q(x_n).reshape(B, C, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)  # B H C D

        # x_1 = self.reorder_channel(x_n)
        # x_2 = x_1

        x_1 = self.conv_region[0](x_n.permute(0, 2, 1)).permute(0, 2, 1)  # B d_model <-> C/8
        x_2 = self.conv_region[1](x_n.permute(0, 2, 1)).permute(0, 2, 1)  # B d_model <-> C/4

        kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads//2, D//self.num_heads).permute(2, 0, 3, 1, 4)
        kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads//2, D//self.num_heads).permute(2, 0, 3, 1, 4)
        k1, v1 = kv1[0], kv1[1]  # B H/2 C/8 D/H
        k2, v2 = kv2[0], kv2[1]  # B H/2 C/4 D/H

        attn1 = (q[:, :self.num_heads // 2] @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)  # B H/2 C C/8
        attn1_d = self.dropout(attn1)
        h1 = (attn1_d @ v1).transpose(1, 2).reshape(B, C, D // 2)  # B H/2 C D/H  ->  B C D/2

        attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)  # B H/2 C C/8
        attn2_d = self.dropout(attn2)
        h2 = (attn2_d @ v2).transpose(1, 2).reshape(B, C, D // 2)  # B H/2 C D/H  ->  B C D/2

        h = torch.cat((h1, h2), dim=-1)
        out = self.post(h+x) if self.use_res else self.post(h)
        return out, attn1, attn2

class MTTformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.period_list = config['period']
        self.d_base = config['d_base']
        d_model = self.d_base * (len(self.period_list) + 1)
        num_heads = self.d_base
        dropout = config['dropout']
        c_in = config['Data_shape'][1] + 1 if config['dataset'] == 'ASA' else config['Data_shape'][1]
        # c_in = config['Data_shape'][1]

        self.token_embedding = TokenEmbedding(config)
        self.cross_attn = CrossAttenBlock(config, d_model, num_heads, dropout=dropout)
        self.reduconv = RegionConvBlock(d_model, kernel_size=c_in, padding='valid', bias=False)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(d_model, 2)

    def forward(self, x):

        x_t = self.token_embedding(x)
        x_s, attn1, attn2 = self.cross_attn(x_t)
        x_g = self.reduconv(x_s.permute(0, 2, 1))

        out = self.flatten(x_g)
        out = self.out(out)

        return out, attn1, attn2