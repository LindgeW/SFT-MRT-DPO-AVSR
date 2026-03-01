import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBiModalDecoderLayer(nn.Module):
    r"""TransformerBiModalDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multi-head attention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise, it's done after.
            Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.
    """
    __constants__ = ['norm_first']  # 明确指出这些变量的值在模型执行期间是不变的，这包括超参数、常数等，它们在模型训练和推理过程中不会被更新

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=F.relu,
                 layer_norm_eps=1e-5,
                 batch_first=False,
                 norm_first=False,
                 bias=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias)

        self.aud_multi_head_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                         bias=bias)
        self.vid_multi_head_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                         bias=bias)

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        self.fc_av = nn.Linear(d_model * 2, d_model)   # fuse av and va
        # self.fc_a = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        # self.fc_v = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.LayerNorm(d_model))

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self,
                tgt,
                audio_memory=None,
                video_memory=None,
                tgt_mask=None,    #
                memory_mask=None,
                tgt_key_padding_mask=None,   #
                audio_memory_key_padding_mask=None,   #
                video_memory_key_padding_mask=None,   #
                tgt_is_causal=False,
                memory_is_causal=False):
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            tgt: the sequence to the decoder layer (required).
            audio/video_memory: the sequence from the last layer of the audio/video encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            audio/video_memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.
        Shape:  .
        """
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            norm_x = self.norm2(x)
            if audio_memory is not None and video_memory is not None:
                aud_x = self._aud_mha_block(norm_x, audio_memory, memory_mask, audio_memory_key_padding_mask, memory_is_causal)
                vid_x = self._vid_mha_block(norm_x, video_memory, memory_mask, video_memory_key_padding_mask, memory_is_causal)
            elif video_memory is not None:
                aud_x = torch.zeros_like(x)
                vid_x = self._vid_mha_block(norm_x, video_memory, memory_mask, video_memory_key_padding_mask, memory_is_causal)
            else:
                aud_x = self._aud_mha_block(norm_x, audio_memory, memory_mask, audio_memory_key_padding_mask, memory_is_causal)
                vid_x = torch.zeros_like(x)
            x = x + self.fc_av(torch.cat([aud_x, vid_x], dim=-1))
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            if audio_memory is not None and video_memory is not None:
                aud_x = self._aud_mha_block(x, audio_memory, memory_mask, audio_memory_key_padding_mask, memory_is_causal)
                vid_x = self._vid_mha_block(x, video_memory, memory_mask, video_memory_key_padding_mask, memory_is_causal)
            elif video_memory is not None:
                aud_x = torch.zeros_like(x)
                vid_x = self._vid_mha_block(x, video_memory, memory_mask, video_memory_key_padding_mask, memory_is_causal)
            else:
                aud_x = self._aud_mha_block(x, audio_memory, memory_mask, audio_memory_key_padding_mask, memory_is_causal)
                vid_x = torch.zeros_like(x)
            x = self.norm2(x + self.fc_av(torch.cat([aud_x, vid_x], dim=-1)))
            x = self.norm3(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x,
                  attn_mask,
                  key_padding_mask,
                  is_causal=False):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           # is_causal=is_causal,    # torch>2.0引入的
                           need_weights=False)[0]
        return self.dropout1(x)

    # multi-head attention block
    def _aud_mha_block(self, x, mem,
                       attn_mask,
                       key_padding_mask,
                       is_causal=False):
        x = self.aud_multi_head_attn(x, mem, mem,
                                     attn_mask=attn_mask,
                                     key_padding_mask=key_padding_mask,
                                     # is_causal=is_causal,   # torch>2.0引入的
                                     need_weights=False)[0]
        return self.dropout2(x)

    def _vid_mha_block(self, x, mem,
                       attn_mask,
                       key_padding_mask,
                       is_causal=False):
        x = self.vid_multi_head_attn(x, mem, mem,
                                     attn_mask=attn_mask,
                                     key_padding_mask=key_padding_mask,
                                     # is_causal=is_causal,   # torch>2.0引入的
                                     need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class TransformerBiModalDecoder(nn.Module):
    r"""TransformerBiModalDecoder is a stack of N decoder layers.
    Args:
        decoder_layer: an instance of the TransformerBiModalDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self,
                 decoder_layer,  # TransformerBiModalDecoderLayer
                 num_layers,
                 norm=None):
        super().__init__()
        # torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                tgt,
                audio_memory=None,
                video_memory=None,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                audio_memory_key_padding_mask=None,
                video_memory_key_padding_mask=None,
                tgt_is_causal=None,
                memory_is_causal=False):
        r"""Pass the inputs (and mask) through the decoder layer in turn.
        Args:
            tgt: the sequence to the decoder (required).
            audio_video/memory: the sequence from the last layer of the audio/video encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            audio/video_memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.
        Shape:
            - src: :math:`(S, E)` for unbatched input, :math:`(S, N, E)` if `batch_first=False` or
              `(N, S, E)` if `batch_first=True`.
            - tgt: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.
            - src_mask: :math:`(S, S)` or :math:`(N\cdot\text{num\_heads}, S, S)`.
            - tgt_mask: :math:`(T, T)` or :math:`(N\cdot\text{num\_heads}, T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(T)` for unbatched input otherwise :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.

            Note: [src/tgt/memory]_mask ensures that position :math:`i` is allowed to attend the unmasked
            positions. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            - output: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.

            where :math:`S` is the source sequence length, :math:`T` is the target sequence length, :math:`N` is the
            batch size, :math:`E` is the feature number
        """
        output = tgt
        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)
        for mod in self.layers:
            output = mod(output,
                         audio_memory,
                         video_memory,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         audio_memory_key_padding_mask=audio_memory_key_padding_mask,
                         video_memory_key_padding_mask=video_memory_key_padding_mask,
                         tgt_is_causal=tgt_is_causal,
                         memory_is_causal=memory_is_causal)
        if self.norm is not None:
            output = self.norm(output)
        return output


def _get_seq_len(src, batch_first):
    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _generate_square_subsequent_mask(sz, device=None, dtype=None, ):
    r"""Generate a square causal mask for the sequence.
    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    return torch.triu(torch.full((sz, sz), float('-inf'), dtype=dtype, device=device), diagonal=1, )


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _detect_is_causal_mask(mask, is_causal=None, size=None, ) -> bool:
    """Return whether the given attention mask is causal.
    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,
    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)
    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(sz, device=mask.device, dtype=mask.dtype)
        # Do not use `torch.equal` so we handle batched masks by broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False
    return make_causal
