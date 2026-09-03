import torch
from typing import Optional, Union, List

from nemo.collections.asr.modules.rnnt import RNNTJoint
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.collections.asr.parts.submodules.jasper import init_weights
from nemo.utils import logging
try:
    import k2
except ImportError:
    logging.warning("k2 is not installed, RNN-T training will be disabled")

_ACTIVATIONS = {
    'silu':    lambda: torch.nn.SiLU(inplace=True),
    'swish':   lambda: torch.nn.SiLU(inplace=True),
    'relu':    lambda: torch.nn.ReLU(inplace=True),
    'tanh':    lambda: torch.nn.Tanh(),
    'sigmoid': lambda: torch.nn.Sigmoid(),
    'gelu':    lambda: torch.nn.GELU(),
}


def _make_activation(name):
    name = name.lower()
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {name}. Choose from {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]()


class ProjHead(NeuralModule, Exportable):
    """Configurable MLP projection head:

        encoder_output (B, D, T)
          -> transpose to (B, T, D)
          -> [Linear(D, h_1) -> act] -> ... -> [Linear(h_{k-1}, h_k) -> act]
          -> Linear(h_k, num_classes)   # named `decoder_layers`, bias=False

    Configuration via ``hidden_dims`` (list of ints):
        []         -> single Linear (legacy SimpleProj behavior)
        [H]        -> Linear -> act -> Linear (legacy TwoLayerProj behavior)
        [H1, H2]   -> three Linears, etc.

    The final Linear is always named ``decoder_layers`` (bias=False) so that
    existing CTC-init scripts and SimpleProj checkpoints continue to load —
    when ``hidden_dims=[]``, the entire state_dict matches the old SimpleProj
    keys exactly.

    Pre-layers live under ``pre`` (an ``nn.Sequential``). With ``hidden_dims=[]``
    this is an empty Sequential acting as identity (no params, no state_dict keys).

    Args:
        feat_in:     input feature dim (D).
        num_classes: output vocab size.
        hidden_dims: list of intermediate hidden dims (default: []).
        activation:  one of [silu, swish, relu, tanh, sigmoid, gelu] (default: silu).
        tie_weights: if True, the final Linear's weight is provided externally to
                     forward() (used for weight-tied LM heads).
    """

    def __init__(self, feat_in, num_classes, hidden_dims=None, activation='silu',
                 init_mode='xavier_uniform', vocabulary=None, tie_weights=False, init_scale=0.25):
        super().__init__()
        self.__vocabulary = vocabulary
        self._feat_in = feat_in
        self._num_classes = num_classes
        self.tie_weights = tie_weights

        hidden_dims = list(hidden_dims) if hidden_dims else []
        self._hidden_dims = hidden_dims

        pre_layers = []
        prev = feat_in
        for h in hidden_dims:
            pre_layers.append(torch.nn.Linear(prev, h))
            pre_layers.append(_make_activation(activation))
            prev = h
        self.pre = torch.nn.Sequential(*pre_layers)  # empty Sequential is a valid identity

        if not self.tie_weights:
            self.decoder_layers = torch.nn.Linear(prev, num_classes, bias=False)

        self.apply(lambda x: init_weights(x, mode=init_mode))
        if not self.tie_weights and init_scale != 1.0:
            self.decoder_layers.weight.data.mul_(init_scale)
        # to change, requires running ``model.temperature = T`` explicitly
        self.temperature = 1.0

    def forward(self, encoder_output, return_logits=False, return_softmax=True, weights=None):
        assert return_logits or return_softmax, "Either return_logits or return_softmax must be True"
        # (B, D, T) -> (B, T, D) -> pre -> (B, T, H_last) -> final Linear -> (B, T, V)
        x = self.pre(encoder_output.transpose(1, 2))
        if not self.tie_weights:
            logits = self.decoder_layers(x)
        else:
            assert weights is not None, "weights must be provided if tie_weights is True"
            logits = torch.nn.functional.linear(x, weights)
        if return_softmax:
            # Under Lightning's bf16-mixed autocast, log_softmax is on the fp32-promotion
            # list — autocast upcasts internally and emits fp32. Explicit .float() is
            # redundant here and only adds a transient fp32 logits copy at forward peak.
            softmax = torch.nn.functional.log_softmax(logits / self.temperature, dim=-1)
            if return_logits:
                return logits, softmax
            else:
                return softmax
        else:
            return logits

    @property
    def num_classes_with_blank(self):
        return self._num_classes

class DistilTimeUpsampler(torch.nn.Module):

    def __init__(self, d_model: int, ratio: int, activation: str = "gelu"):
        super().__init__()
        self.ratio = ratio
        self.d_model = d_model
        act = {"gelu": torch.nn.GELU, "silu": torch.nn.SiLU, "swish": torch.nn.SiLU,
               "relu": torch.nn.ReLU}[activation.lower()]
        self.act = act()
        self.proj = torch.nn.Linear(d_model, ratio * d_model)

    def forward(self, x):                       # x: (B, D, T)
        x = x.transpose(1, 2)                   # (B, T, D)
        x = self.proj(self.act(x))              # (B, T, ratio*D)
        b, t, _ = x.shape
        x = x.reshape(b, t, self.ratio, self.d_model).reshape(b, t * self.ratio, self.d_model)
        return x.transpose(1, 2)                # (B, D, T*ratio)


class PrunedRNNTJoint(RNNTJoint):
    def _joint_net_modules(self, num_classes, pred_n_hidden, enc_n_hidden, joint_n_hidden, activation, dropout):
        """Override NeMo's _joint_net_modules to:

        1. Skip the encoder / predictor projection (use nn.Identity) when its input
           dim already matches joint_n_hidden. Saves a Linear's worth of params,
           compute, and activation memory whenever the dims happen to line up.
        2. Accept 'silu' (a.k.a. swish) in addition to NeMo's [relu, tanh, sigmoid].
           SiLU is the default modern transformer activation (used by Llama, Qwen,
           PaLM, etc.) — typically matches or slightly beats ReLU/Tanh on RNN-T joints.
        """
        if pred_n_hidden == joint_n_hidden:
            pred = torch.nn.Identity()
        else:
            pred = torch.nn.Linear(pred_n_hidden, joint_n_hidden)
        if enc_n_hidden == joint_n_hidden:
            enc = torch.nn.Identity()
        else:
            enc = torch.nn.Linear(enc_n_hidden, joint_n_hidden)

        activation = activation.lower()
        if activation == 'relu':
            act = torch.nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            act = torch.nn.Sigmoid()
        elif activation == 'tanh':
            act = torch.nn.Tanh()
        elif activation == 'silu' or activation == 'swish':
            act = torch.nn.SiLU(inplace=True)
        else:
            raise ValueError(
                "Unsupported activation for joint step - please pass one of "
                "[relu, sigmoid, tanh, silu]"
            )

        layers = (
            [act]
            + ([torch.nn.Dropout(p=dropout)] if dropout else [])
            + [torch.nn.Linear(joint_n_hidden, num_classes)]
        )
        return pred, enc, torch.nn.Sequential(*layers)

    def forward(self, f, g, project_input=True):
        if project_input:
            return self.joint_after_projection(self.project_encoder(f), self.project_prednet(g), log_softmax=not self.training)
        else:
            return self.joint_after_projection(f, g, log_softmax=not self.training)

    def joint_after_projection(self, f: torch.Tensor, g: torch.Tensor, log_softmax: bool = True) -> torch.Tensor:
        # f, g -> [B, T, R, H] for training, [B, 1, H] for transcribing
        if f.ndim == 3:
            assert g.ndim == 3
            f = f.unsqueeze(dim=1)  # (B, 1, 1, H)
            g = g.unsqueeze(dim=1)  # (B, 1, 1, H)
        assert g.ndim == 4
        assert f.ndim == 4

        inp = f + g  # [B, T, R, H] or [B, 1, 1, H]

        del f, g

        res = self.joint_net(inp)  # [B, T, R, V + 1] or [B, 1, 1, V + 1]

        del inp


        res = res / self.temperature
        if log_softmax:
            res = res.log_softmax(dim=-1)

        return res

    def forward_fused_loss(
        self,
        encoder_outputs: torch.Tensor,
        decoder_outputs: Optional[torch.Tensor],
        simple_am: Optional[torch.Tensor] = None,
        simple_lm: Optional[torch.Tensor] = None,
        am_only_scale: float = 0.0,
        lm_only_scale: float = 0.25,
        s_range: int = 5,
        delay_penalty: float = 0.0,
        blank_symbol: int = 0,
        encoder_lengths: Optional[torch.Tensor] = None,
        transcripts: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        target_start: Optional[torch.Tensor] = None,
        target_end: Optional[torch.Tensor] = None,
        compute_wer: bool = False,
    ) -> Union[torch.Tensor, List[Optional[torch.Tensor]]]:
        # encoder = (B, D, T)
        # decoder = (B, D, U) if passed, else None
        encoder_outputs = encoder_outputs.transpose(1, 2)  # (B, T, D)

        if decoder_outputs is not None:
            decoder_outputs = decoder_outputs.transpose(1, 2)  # (B, U, D)

        if not self._fuse_loss_wer:
            if decoder_outputs is None:
                raise ValueError(
                    "decoder_outputs passed is None, and `fuse_loss_wer` is not set. "
                    "decoder_outputs can only be None for fused step!"
                )

            out = self.forward(encoder_outputs, decoder_outputs)  # [B, T, U, V + 1]
            return out

        else:
            # At least the loss module must be supplied during fused joint
            if self._wer is None:
                raise ValueError("`fuse_loss_wer` flag is set, but `wer` modules were not provided! ")

            # When using fused joint step, both encoder and transcript lengths must be provided
            if (encoder_lengths is None) or (targets is None) or (target_start is None) or (target_end is None):
                raise ValueError(
                    "`fuse_loss_wer` is set, therefore encoder and target lengths " "must be provided as well!"
                )
            if simple_lm is not None and simple_am is not None:
                # Re-index the RNN-T symbol axis to start at the first TRANSCRIPT token
                # (boundary s_begin = 0). The decoder is causal over [prompt ; transcript],
                # so the predictor states for the transcript region already encode the prompt
                # as left-context; we just gather them. Encoding the prompt skip via a
                # nonzero s_begin instead breaks k2 pruning: the kept window at frame 0 is
                # [0, s_range) but the boundary forces the path to start at (s_begin, 0), so
                # whenever s_begin >= s_range no path survives the pruned lattice -> +inf.
                B = transcripts.shape[0]
                target_lens = target_end - target_start          # (B,) transcript lengths U
                # `targets` is already the padded transcript tokens, i.e. exactly
                # context[target_start : target_end], so it IS the symbol sequence
                # (s = 0..U-1) — no per-sample gather and no .max().item() host sync needed.
                symbols = targets
                max_u = targets.shape[1]
                # predictor states s = 0..U  <-  the causal decoder states that predict
                # transcript tokens 0..U-1, plus the post-last state for the termination
                # transition: positions target_start-1 .. target_start-1+max_u (length max_u+1).
                ar_lm = torch.arange(max_u + 1, device=transcripts.device)
                lm_idx = ((target_start - 1).unsqueeze(1) + ar_lm).clamp_(0, decoder_outputs.shape[1] - 1)
                simple_lm = simple_lm.gather(1, lm_idx.unsqueeze(-1).expand(-1, -1, simple_lm.shape[-1]))
                decoder_outputs = decoder_outputs.gather(1, lm_idx.unsqueeze(-1).expand(-1, -1, decoder_outputs.shape[-1]))

                boundary = torch.zeros((B, 4), dtype=torch.int64, device=encoder_outputs.device)
                boundary[:, 2] = target_lens                     # s_end = U  (s_begin stays 0)
                boundary[:, 3] = encoder_lengths                 # t_end = T (acoustic side full)

                # k2 RNNT kernels are not safe under bf16/fp16 autocast even with
                # fp32-cast inputs — autocast can still affect internal ops. Disable
                # autocast around k2 calls (matches icefall's torch_autocast guard).
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                        lm=simple_lm.float(),
                        am=simple_am.float(),
                        symbols=symbols,
                        termination_symbol=blank_symbol,
                        lm_only_scale=lm_only_scale,
                        am_only_scale=am_only_scale,
                        delay_penalty=delay_penalty,
                        boundary=boundary,
                        reduction='none',
                        return_grad=True,
                    )

                    ranges = k2.get_rnnt_prune_ranges(
                        px_grad=px_grad,
                        py_grad=py_grad,
                        boundary=boundary,
                        s_range=s_range,
                    )

                enc_pruned, dec_pruned = k2.do_rnnt_pruning(
                    am=self.project_encoder(encoder_outputs),
                    lm=self.project_prednet(decoder_outputs),
                    ranges=ranges,
                )
                joint = self.forward(enc_pruned, dec_pruned, project_input=False)
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    rnnt_loss = k2.rnnt_loss_pruned(
                        logits=joint.float(),
                        symbols=symbols,
                        ranges=ranges,
                        termination_symbol=blank_symbol,
                        boundary=boundary,
                        delay_penalty=delay_penalty,
                        reduction='none',
                    )
                del joint, ranges, boundary

                # Per-sample finite check (icefall style): compute full batch first,
                # then mask out non-finite samples. Use multiplication (not indexing)
                # so the computational graph stays connected on all ranks — disconnected
                # zero tensors cause DDP to hang (find_unused_parameters=False means
                # DDP waits for every parameter's all-reduce hook to fire).
                # Independent masks per loss: a sample with finite simple_loss but
                # non-finite rnnt_loss (e.g. s_range pruning missed the true alignment)
                # still contributes to the simple/CTC path, and vice versa.
                simple_finite = torch.isfinite(simple_loss)
                rnnt_finite   = torch.isfinite(rnnt_loss)
                if not (simple_finite.all() and rnnt_finite.all()):
                    logging.warning(
                        f"non-finite samples: simple={(~simple_finite).sum().item()}/{simple_loss.shape[0]}, "
                        f"rnnt={(~rnnt_finite).sum().item()}/{rnnt_loss.shape[0]}"
                    )
                # nan_to_num first: inf * 0 = nan in IEEE 754, which would survive sum()
                simple_loss = simple_loss.nan_to_num(0.0) * simple_finite
                rnnt_loss   = rnnt_loss.nan_to_num(0.0) * rnnt_finite
                simple_tokens = (target_lens * simple_finite).sum().clamp(min=1)
                rnnt_tokens   = (target_lens * rnnt_finite).sum().clamp(min=1)
                simple_loss = simple_loss.sum() / simple_tokens
                rnnt_loss   = rnnt_loss.sum() / rnnt_tokens
            else:
                simple_loss = None
                rnnt_loss = None

            # Update WER for sub batch
            if compute_wer:
                encoder_outputs = encoder_outputs.transpose(1, 2)  # [B, T, D] -> [B, D, T]
                encoder_outputs = encoder_outputs.detach()
                targets = targets.detach()

                # Update WER on each process without syncing
                if self.training:
                    original_sync = self.wer._to_sync
                    self.wer._to_sync = False

                # Seed the greedy RNN-T decoder with the same prompt prefix the
                # model was trained on: context[:, :target_start] = [bos, <language>, ...].
                # The loss aligns the transcript starting from the post-prompt predictor
                # state (s=0 is the causal decoder state at target_start-1), so the joint
                # is only ever trained on prompt-conditioned predictor states; decoding
                # from bos alone is out-of-distribution and emits only blanks. Slice to
                # the COMMON prefix (min target_start) so no transcription token is
                # ever leaked for variable-length prompts.
                prompt_len = int(target_start.min().item())
                prompt_ids = transcripts[:, :prompt_len] if prompt_len > 0 else None
                self.wer.update(
                    predictions=encoder_outputs,
                    predictions_lengths=encoder_lengths,
                    targets=targets,
                    targets_lengths=target_end - target_start,
                    input_ids=prompt_ids,
                )
                # Sync and all_reduce on all processes, compute global WER
                wer, wer_num, wer_denom = self.wer.compute()
                self.wer.reset()

                if self.training:
                    self.wer._to_sync = original_sync
            else:
                wer = None
                wer_num = None
                wer_denom = None

            return simple_loss, rnnt_loss, wer, wer_num, wer_denom
