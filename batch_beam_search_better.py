from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence



def tile(x: Tensor | tuple[Tensor, ...], count: int, dim: int = 0) -> Tensor | tuple[Tensor, ...]:
    """
    Repeat a tensor along one dimension for beam expansion.

    This is equivalent to duplicating each item ``count`` times along ``dim``.
    """
    if isinstance(x, tuple):
        return tuple(tile(item, count, dim) for item in x)
    return x.repeat_interleave(count, dim=dim).contiguous()


def _length_penalty(length: int, alpha: float) -> float:
    if alpha < 0:
        return 1.0
    return ((5.0 + float(length)) / 6.0) ** alpha


def _trim_to_eos(sequence: Tensor, eos_index: int) -> Tensor:
    eos_positions = torch.nonzero(sequence == eos_index, as_tuple=False)
    if eos_positions.numel() == 0:
        return sequence
    first_eos = int(eos_positions[0].item())
    return sequence[: first_eos + 1]


def _extract_step_logits(logits: Tensor, batch_beam_size: int) -> Tensor:
    """
    Return logits for the newest decoding step.

    The expected decoder output is ``[batch_beam, time, vocab]``. For minimum
    compatibility, ``[batch_beam, vocab]`` is also accepted.
    """
    if logits.dim() == 2:
        return logits
    if logits.dim() < 2:
        raise ValueError(f"Decoder logits must have at least 2 dimensions, got {tuple(logits.shape)}.")

    if logits.size(0) != batch_beam_size:
        logits = logits.reshape(batch_beam_size, -1, logits.size(-1))
    elif logits.dim() != 3:
        logits = logits.reshape(batch_beam_size, -1, logits.size(-1))

    return logits[:, -1, :]


def beam_decode(
    decoder: Any,
    encoder_output: Tensor,
    src_lens: Tensor,
    bos_index: int,
    eos_index: int,
    beam_size: int = 5,
    max_output_length: int = 100,
    alpha: float = 0.6,   # -1, 0, 0.7, 1.0
    n_best: int = 1,
):
    """
    Beam search for a Transformer decoder.

    Compatibility notes:
    - Keeps the current decoder call pattern:
      ``decoder(decoder_input, encoder_output, tgt_lens=..., src_lens=...)``
    - Keeps the legacy return style:
      ``n_best == 1`` -> padded Tensor ``[batch, dec_len]``
      ``n_best > 1`` -> ``(predictions, scores)``

    Search notes:
    - Finished hypotheses are never expanded again.
    - Length penalty is used only for ranking, while cumulative raw log-prob is
      preserved internally.
    """
    assert beam_size > 0, "Beam size must be > 0."
    assert max_output_length > 0, "max_output_length must be > 0."
    assert n_best > 0, "n_best must be > 0."
    assert n_best <= beam_size, f"Can only return up to {beam_size} best hypotheses."

    device = encoder_output.device
    batch_size = int(src_lens.size(0))

    encoder_output = tile(encoder_output.contiguous(), beam_size, dim=0)
    src_lens = tile(src_lens.contiguous(), beam_size, dim=0)

    alive_seq = torch.full(
        (batch_size * beam_size, 1),
        bos_index,
        dtype=torch.long,
        device=device,
    )

    alive_log_probs = torch.full(
        (batch_size, beam_size),
        float("-inf"),
        dtype=encoder_output.dtype,
        device=device,
    )
    alive_log_probs[:, 0] = 0.0

    active_batch_ids = torch.arange(batch_size, dtype=torch.long, device=device)
    finished: list[list[tuple[float, Tensor]]] = [[] for _ in range(batch_size)]

    for step in range(max_output_length):
        active_batch_size = int(active_batch_ids.size(0))
        if active_batch_size == 0:
            break

        decoder_input = alive_seq
        tgt_lens = torch.full(
            (decoder_input.size(0),),
            decoder_input.size(1),
            dtype=torch.long,
            device=device,
        )

        step_logits = _extract_step_logits(
            decoder(decoder_input, encoder_output, tgt_lens=tgt_lens, src_lens=src_lens),
            decoder_input.size(0),
        )

        log_probs = F.log_softmax(step_logits, dim=-1)
        vocab_size = int(log_probs.size(-1))

        raw_candidate_log_probs = log_probs.view(active_batch_size, beam_size, vocab_size)
        raw_candidate_log_probs = raw_candidate_log_probs + alive_log_probs.unsqueeze(-1)

        next_length = decoder_input.size(1)
        penalty = _length_penalty(next_length, alpha)
        ranked_candidate_scores = raw_candidate_log_probs / penalty

        flat_ranked_scores = ranked_candidate_scores.reshape(active_batch_size, beam_size * vocab_size)
        flat_raw_log_probs = raw_candidate_log_probs.reshape(active_batch_size, beam_size * vocab_size)
        beam_offsets = torch.arange(0, active_batch_size * beam_size, step=beam_size, device=device)

        sample_done = torch.zeros(active_batch_size, dtype=torch.bool, device=device)
        next_alive_seq: list[Tensor] = []
        next_alive_log_probs: list[Tensor] = []
        next_encoder_indices: list[Tensor] = []
        next_batch_ids: list[int] = []

        if step + 1 == max_output_length:
            final_scores, final_flat_ids = flat_ranked_scores.topk(beam_size, dim=-1)
            final_parent = torch.div(final_flat_ids, vocab_size, rounding_mode="floor")
            final_tokens = final_flat_ids.fmod(vocab_size)
            final_parent_indices = final_parent + beam_offsets.unsqueeze(1)

            final_prefixes = alive_seq.index_select(0, final_parent_indices.reshape(-1))
            final_prefixes = final_prefixes.view(active_batch_size, beam_size, -1)
            final_sequences = torch.cat([final_prefixes, final_tokens.unsqueeze(-1)], dim=-1)

            for i in range(active_batch_size):
                batch_id = int(active_batch_ids[i].item())
                for j in range(beam_size):
                    score = float(final_scores[i, j].item())
                    prediction = _trim_to_eos(final_sequences[i, j, 1:], eos_index).detach().clone()
                    finished[batch_id].append((score, prediction))
                sample_done[i] = True
        else:
            eos_scores = ranked_candidate_scores[:, :, eos_index]
            top_finished_scores, top_finished_parents = eos_scores.topk(beam_size, dim=-1)
            top_finished_raw_log_probs = raw_candidate_log_probs[:, :, eos_index].gather(1, top_finished_parents)
            top_finished_parent_indices = top_finished_parents + beam_offsets.unsqueeze(1)

            top_finished_prefixes = alive_seq.index_select(0, top_finished_parent_indices.reshape(-1))
            top_finished_prefixes = top_finished_prefixes.view(active_batch_size, beam_size, -1)
            eos_column = torch.full(
                (active_batch_size, beam_size, 1),
                eos_index,
                dtype=torch.long,
                device=device,
            )
            top_finished_sequences = torch.cat([top_finished_prefixes, eos_column], dim=-1)

            non_eos_scores = ranked_candidate_scores.clone()
            non_eos_scores[:, :, eos_index] = float("-inf")
            flat_non_eos_scores = non_eos_scores.reshape(active_batch_size, beam_size * vocab_size)

            top_alive_scores, top_alive_flat_ids = flat_non_eos_scores.topk(beam_size, dim=-1)
            top_alive_raw_log_probs = flat_raw_log_probs.gather(1, top_alive_flat_ids)
            top_alive_parents = torch.div(top_alive_flat_ids, vocab_size, rounding_mode="floor")
            top_alive_tokens = top_alive_flat_ids.fmod(vocab_size)
            top_alive_parent_indices = top_alive_parents + beam_offsets.unsqueeze(1)

            top_alive_prefixes = alive_seq.index_select(0, top_alive_parent_indices.reshape(-1))
            top_alive_prefixes = top_alive_prefixes.view(active_batch_size, beam_size, -1)
            top_alive_sequences = torch.cat([top_alive_prefixes, top_alive_tokens.unsqueeze(-1)], dim=-1)

            for i in range(active_batch_size):
                batch_id = int(active_batch_ids[i].item())

                for j in range(beam_size):
                    raw_finished = top_finished_raw_log_probs[i, j]
                    if torch.isfinite(raw_finished):
                        score = float(top_finished_scores[i, j].item())
                        prediction = top_finished_sequences[i, j, 1:].detach().clone()
                        finished[batch_id].append((score, prediction))

                finished[batch_id].sort(key=lambda item: item[0], reverse=True)

                valid_alive_mask = torch.isfinite(top_alive_scores[i])
                sample_alive_scores = top_alive_scores[i][valid_alive_mask]
                sample_alive_log_probs = top_alive_raw_log_probs[i][valid_alive_mask]
                sample_alive_sequences = top_alive_sequences[i][valid_alive_mask]
                sample_alive_parents = top_alive_parent_indices[i][valid_alive_mask]

                if len(finished[batch_id]) >= n_best:
                    nth_finished_score = finished[batch_id][n_best - 1][0]
                    if sample_alive_scores.numel() == 0 or float(sample_alive_scores[0].item()) <= nth_finished_score:
                        sample_done[i] = True
                        continue

                if sample_alive_scores.numel() == 0:
                    sample_done[i] = True
                    continue

                next_alive_seq.append(sample_alive_sequences)
                next_alive_log_probs.append(sample_alive_log_probs)
                next_encoder_indices.append(sample_alive_parents)
                next_batch_ids.append(batch_id)

        if bool(sample_done.all()):
            break

        alive_seq = torch.cat(next_alive_seq, dim=0)
        alive_log_probs = torch.stack(next_alive_log_probs, dim=0)
        select_indices = torch.cat(next_encoder_indices, dim=0)
        encoder_output = encoder_output.index_select(0, select_indices)
        src_lens = src_lens.index_select(0, select_indices)
        active_batch_ids = torch.tensor(next_batch_ids, dtype=torch.long, device=device)

    for batch_id in range(batch_size):
        finished[batch_id].sort(key=lambda item: item[0], reverse=True)
        if not finished[batch_id]:
            fallback = torch.full((1,), eos_index, dtype=torch.long, device=device)
            finished[batch_id].append((float("-inf"), fallback))

    if n_best == 1:
        best_predictions = [batch_finished[0][1] for batch_finished in finished]
        return pad_sequence(best_predictions, batch_first=True, padding_value=eos_index)

    final_outputs: list[list[Tensor]] = []
    final_scores: list[list[float]] = []
    for batch_finished in finished:
        chosen = batch_finished[:n_best]
        final_outputs.append([prediction.detach().cpu() for _, prediction in chosen])
        final_scores.append([score for score, _ in chosen])

    return final_outputs, final_scores
