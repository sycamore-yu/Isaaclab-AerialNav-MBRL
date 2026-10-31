"""Keep Torch/Warp stream ownership stable across captured task stages."""

import torch
import warp as wp

_EXTERNAL_STREAMS: dict[tuple[str, int], tuple[torch.cuda.Stream, wp.Stream]] = {}


def torch_warp_stream(device):
    """Return the current Torch stream without re-wrapping an owned Warp stream.

    In Warp 1.17, destroying a temporary external wrapper unregisters its CUDA
    handle. Re-wrapping an existing Warp stream can therefore break a later
    graph capture on that stream. Reuse the current owner when the handles
    match; retain both owners for genuinely external Torch streams.
    """
    if torch.device(device).type != "cuda":
        return None
    torch_stream = torch.cuda.current_stream(device)
    warp_stream = wp.get_stream(str(torch_stream.device))
    if warp_stream.cuda_stream == torch_stream.cuda_stream:
        return warp_stream
    key = (str(torch_stream.device), torch_stream.cuda_stream)
    if key not in _EXTERNAL_STREAMS:
        _EXTERNAL_STREAMS[key] = (torch_stream, wp.stream_from_torch(torch_stream))
    return _EXTERNAL_STREAMS[key][1]
