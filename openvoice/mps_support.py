import torch


@torch.no_grad()
def convt1d_chunked(deconv: torch.nn.ConvTranspose1d, x: torch.Tensor, chunk: int = 32768, overlap: int = 2):
    N, C, L = x.shape
    outs = []
    start = 0
    while start < L:
        end = min(L, start + chunk)
        left_ctx = max(0, start - overlap)
        right_ctx = min(L, end + overlap)
        x_piece = x[..., left_ctx:right_ctx]
        y_piece = deconv(x_piece)
        left_cut = start - left_ctx
        right_cut = y_piece.shape[-1] - (right_ctx - end)
        if right_cut == 0:
            y_valid = y_piece[..., left_cut:]
        else:
            y_valid = y_piece[..., left_cut:right_cut]
        outs.append(y_valid)
        start = end
    return torch.cat(outs, dim=-1)


@torch.no_grad()
def conv1d_chunked(conv: torch.nn.Conv1d, x: torch.Tensor, chunk: int = 32768, overlap: int = 2):
    N, C, L = x.shape
    ys = []
    start = 0
    while start < L:
        end = min(L, start + chunk)
        left_ctx = max(0, start - overlap)
        right_ctx = min(L, end + overlap)
        x_piece = x[..., left_ctx:right_ctx]
        y_piece = conv(x_piece)
        left_cut = start - left_ctx
        right_cut = y_piece.shape[-1] - (right_ctx - end)
        if right_cut == 0:
            y_valid = y_piece[..., left_cut:]
        else:
            y_valid = y_piece[..., left_cut:right_cut]
        ys.append(y_valid)
        start = end
    return torch.cat(ys, dim=-1)
