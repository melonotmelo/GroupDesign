import torch


def precompute_freqs_cis_1d(dim: int, x_len: int, x_factor: float = 1.0, base: float = 10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(x_len, device=freqs.device) / x_factor
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def precompute_freqs_cis_2d(dim: int, x_len: int, y_len: int,
                            x_factor: float = 1.0, y_factor: float = 1.0, base: float = 10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, dim/2, 2).float() / dim))

    x_t = torch.arange(x_len, device=freqs.device) / x_factor
    y_t = torch.arange(y_len, device=freqs.device) / y_factor

    x_freqs = torch.outer(x_t, freqs).float()
    y_freqs = torch.outer(y_t, freqs).float()

    x_freqs = x_freqs.unsqueeze(0).repeat(y_len, 1, 1)
    y_freqs = y_freqs.unsqueeze(1).repeat(1, x_len, 1)

    freqs = torch.concat((x_freqs, y_freqs), dim=-1)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    freqs_cis = freqs_cis.view(-1, freqs_cis.shape[-1])

    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
        position_ids: torch.Tensor=None
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    if position_ids is not None:
        freqs_cis = reshape_for_broadcast(freqs_cis[position_ids], xq_)
    else:
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

