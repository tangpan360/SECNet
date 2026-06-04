import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RevIN import RevIN


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.channels = channels
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.layer_norm(x, (self.channels,), self.weight, self.bias, self.eps)
        return x.transpose(1, 2)


class SETCBBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.pre_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.dilated_conv = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.chomp = Chomp1d(padding)
        self.post_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.norm = ChannelLayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.pre_conv(x)
        x = self.dilated_conv(x)
        x = self.chomp(x)
        x = self.post_conv(x)
        x = F.gelu(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x + residual


class CAASD(nn.Module):
    def __init__(self, num_features, seq_len, topk):
        super().__init__()
        self.freq_bins = seq_len // 2 + 1

        scale = 1.0 / (num_features * num_features)
        self.topk = topk
        self.core_weight = nn.Parameter(scale * torch.rand(self.freq_bins, num_features, num_features, dtype=torch.cfloat))
        self.aux_weight = nn.Parameter(scale * torch.rand(self.freq_bins, num_features, num_features, dtype=torch.cfloat))

    def forward(self, x):
        x_ft = torch.fft.rfft(x, dim=1)
        amplitudes = x_ft.abs()
        k = max(1, min(self.topk, amplitudes.size(1)))

        topk_indices = torch.topk(amplitudes, k=k, dim=1).indices
        topk_mask = torch.zeros_like(amplitudes, dtype=torch.bool)
        topk_mask.scatter_(1, topk_indices, True)
        residual_mask = ~topk_mask

        core_ft = x_ft * topk_mask
        aux_ft = x_ft * residual_mask

        core_ft = torch.einsum("bfc,fcd->bfd", core_ft, self.core_weight)
        aux_ft = torch.einsum("bfc,fcd->bfd", aux_ft, self.aux_weight)

        y_core = torch.fft.irfft(core_ft, n=x.size(1), dim=1)
        y_aux = torch.fft.irfft(aux_ft, n=x.size(1), dim=1)
        return y_core, y_aux


class AuxiliaryHighwayNetwork(nn.Module):
    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, pred_len * 2),
            nn.Linear(pred_len * 2, pred_len),
        )

    def forward(self, y_aux):
        return self.net(y_aux.transpose(1, 2))


class LightweightRegressionFusion(nn.Module):
    def __init__(self, patch_num, core_channels, seq_len, pred_len, dropout):
        super().__init__()
        self.core_proj = nn.Linear(patch_num * core_channels, pred_len)
        self.residual_proj = nn.Linear(seq_len, pred_len)
        self.dropout = nn.Dropout(dropout)
        self.fusion_proj = nn.Linear(pred_len, pred_len)

    def forward(self, core_features, residual_input, aux_pred):
        core_pred = self.core_proj(core_features.flatten(start_dim=2))
        residual_pred = self.residual_proj(residual_input)
        fused = self.fusion_proj(self.dropout(F.gelu(core_pred + residual_pred)))
        return (fused + aux_pred).transpose(1, 2)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.patch_num = int((self.seq_len - self.patch_size) / self.stride + 1) + 1

        num_channels = configs.num_channels
        if not num_channels:
            raise ValueError("SECNet requires at least one channel size in num_channels.")

        self.revin_layer = RevIN(num_features=self.enc_in, affine=False, subtract_last=False)
        self.caasd = CAASD(
            num_features=self.enc_in,
            seq_len=self.seq_len,
            topk=configs.topk,
        )
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.segment_embedding = nn.Linear(self.patch_size, configs.d_model)

        blocks = []
        in_channels = configs.d_model
        for i, out_channels in enumerate(num_channels):
            blocks.append(
                SETCBBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=configs.kernel_size,
                    dilation=2 ** i,
                    dropout=configs.dropout,
                )
            )
            in_channels = out_channels
        self.core_encoder = nn.Sequential(*blocks)

        self.aux_highway = AuxiliaryHighwayNetwork(seq_len=self.seq_len, pred_len=self.pred_len)
        self.lrf_head = LightweightRegressionFusion(
            patch_num=self.patch_num,
            core_channels=num_channels[-1],
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            dropout=configs.dropout,
        )

    def _segment_core(self, y_core):
        bsz, _, nvars = y_core.shape
        y_core = y_core.transpose(1, 2)
        y_core = self.padding_patch_layer(y_core)
        y_core = y_core.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        y_core = y_core.reshape(bsz * nvars, self.patch_num, self.patch_size)
        y_core = self.segment_embedding(y_core)
        return y_core.transpose(1, 2), nvars

    def forecast(self, x):
        bsz = x.size(0)
        x = self.revin_layer(x, "norm")

        y_core, y_aux = self.caasd(x)
        core_segments, nvars = self._segment_core(y_core)
        core_features = self.core_encoder(core_segments)
        core_features = core_features.reshape(bsz, nvars, core_features.size(1), core_features.size(2))

        residual_input = x.transpose(1, 2)
        aux_pred = self.aux_highway(y_aux)
        y = self.lrf_head(core_features, residual_input, aux_pred)
        y = self.revin_layer(y, "denorm")
        return y

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            return self.forecast(x_enc)[:, -self.pred_len:, :]
        raise NotImplementedError("SECNet currently supports forecasting tasks only.")
