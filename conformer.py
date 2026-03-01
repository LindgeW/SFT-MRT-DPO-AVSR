import torch
import torch.nn as nn
# https://github.com/YUCHEN005/GILA/blob/59d6f014fd9406ce1f1769633566787028c87ce8/fairseq/fairseq/modules/conformer_layer.py
# https://github.com/sooftware/conformer/tree/main/conformer
# https://pytorch.org/audio/main/_modules/torchaudio/models/conformer.html#Conformer


def _lengths_to_padding_mask(lengths):  # torch.Tensor
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())  
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask


def my_lengths_to_padding_mask(lengths, max_len=None):
    '''
     param:   lengths --- [Batch_size]
     return:  mask --- [Batch_size, max_len]
    '''
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = int(torch.max(lengths).item())
    ids = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    pad_mask = ids >= lengths.unsqueeze(1)    ## padding for True 
    return pad_mask



class _ConvolutionModule(nn.Module):
    r"""Conformer convolution module.

    Args:
        input_dim (int): input dimension.
        num_channels (int): number of depthwise convolution layer input channels.
        depthwise_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        bias (bool, optional): indicates whether to add bias term to each convolution layer. (Default: ``False``)
        use_group_norm (bool, optional): use GroupNorm rather than BatchNorm. (Default: ``False``)
    """
    def __init__(
            self,
            input_dim: int,
            num_channels: int,
            depthwise_kernel_size: int,
            dropout: float = 0.0,
            bias: bool = False,
            use_group_norm: bool = False,
    ):
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")
        self.layer_norm = nn.LayerNorm(input_dim)
        self.sequential = nn.Sequential(
            nn.Conv1d(
                input_dim,
                2 * num_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.GLU(dim=1),
            nn.Conv1d(
                num_channels,
                num_channels,
                kernel_size=depthwise_kernel_size,
                stride=1,
                padding=(depthwise_kernel_size - 1) // 2,
                groups=num_channels,
                bias=bias,
            ),
            nn.GroupNorm(num_groups=1, num_channels=num_channels)
            if use_group_norm
            else nn.BatchNorm1d(num_channels),
            nn.SiLU(),    # Swish
            nn.Conv1d(
                num_channels,
                input_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.Dropout(dropout),
        )

    def forward(self, input):
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, D)`.

        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        x = self.layer_norm(input)
        x = x.transpose(1, 2)   # BTD to BDT
        x = self.sequential(x)
        return x.transpose(1, 2)    # BDT to BTD


class _FeedForwardModule(nn.Module):
    r"""Positionwise feed forward layer.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.SiLU(),  # Swish
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, input):
        r"""
        Args:
            input (torch.Tensor): with shape `(*, D)`.

        Returns:
            torch.Tensor: output, with shape `(*, D)`.
        """
        return self.sequential(input)


class ConformerLayer(nn.Module):
    r"""Conformer layer that constitutes Conformer.

    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_attention_heads (int): number of attention heads.
        depthwise_conv_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
    """
    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.0,
        use_group_norm: bool = False,
    ):
        super().__init__()

        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.self_attn_layer_norm = nn.LayerNorm(input_dim)
        self.self_attn = nn.MultiheadAttention(input_dim, num_attention_heads, dropout=dropout)
        self.self_attn_dropout = nn.Dropout(dropout)

        self.conv_module = _ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
        )

        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = nn.LayerNorm(input_dim)

    def _apply_convolution(self, input):
        residual = input
        input = input.transpose(0, 1)   # TBD to BTD
        input = self.conv_module(input)
        input = input.transpose(0, 1)   # BTD to TBD
        input = residual + input
        return input

    def forward(self, input, key_padding_mask):
        r"""
        Args:
            input (torch.Tensor): input, with shape `(T, B, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.

        Returns:
            torch.Tensor: output, with shape `(T, B, D)`.
        """
        residual = input
        x = self.ffn1(input)
        x = x * 0.5 + residual

        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.self_attn_dropout(x)
        x = x + residual

        x = self._apply_convolution(x)

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x


class Conformer(nn.Module):
    r"""Conformer architecture introduced in
    *Conformer: Convolution-augmented Transformer for Speech Recognition*

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Conformer layer.
        ffn_dim (int): hidden layer dimension of feedforward networks.
        num_layers (int): number of Conformer layers to instantiate.
        depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
    """
    def __init__(
            self,
            input_dim: int,
            num_heads: int,
            ffn_dim: int,
            num_layers: int,
            depthwise_conv_kernel_size: int,
            dropout: float = 0.1,
            use_group_norm: bool = False,
    ):
        super().__init__()
        self.conformer_layers = nn.ModuleList(
            [
                ConformerLayer(
                    input_dim,
                    ffn_dim,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input, lengths=None):
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, input_dim)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor)
                torch.Tensor
                    output frames, with shape `(B, T, input_dim)`
                torch.Tensor
                    output lengths, with shape `(B, )` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """
        #encoder_padding_mask = _lengths_to_padding_mask(lengths)
        encoder_padding_mask = my_lengths_to_padding_mask(lengths, input.shape[1]) if lengths is not None else None
        x = input.transpose(0, 1)   # BTD to TBD
        for layer in self.conformer_layers:
            x = layer(x, encoder_padding_mask)
        return x.transpose(0, 1)    # TBD to BTD


    def forward_pair(self, inp_x, x_lens, inp_y, y_lens):
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, input_dim)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor)
                torch.Tensor
                    output frames, with shape `(B, T, input_dim)`
                torch.Tensor
                    output lengths, with shape `(B, )` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """
        encoder_padding_mask_x = my_lengths_to_padding_mask(x_lens, inp_x.shape[1])
        encoder_padding_mask_y = my_lengths_to_padding_mask(y_lens, inp_y.shape[1])
        encoder_padding_mask = torch.cat([encoder_padding_mask_x, encoder_padding_mask_y], dim=1)
        z = torch.cat([inp_x, inp_y], dim=1)
        z = z.transpose(0, 1)   # BTD to TBD
        for layer in self.conformer_layers:
            z = layer(z, encoder_padding_mask)
        z = z.transpose(0, 1)    # TBD to BTD
        x, y = z[:, :inp_x.size(1)], z[:, inp_x.size(1):]
        return x, y

