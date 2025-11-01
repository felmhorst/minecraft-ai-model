import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(
            self,
            dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    def forward(
            self,
            t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B,1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        half = self.dimensions // 2
        embedding = math.log(10000) / (half - 1)
        embedding = torch.exp(torch.arange(half, device=t.device) * -embedding)
        embedding = t.float() * embedding[None, :]
        embedding = torch.cat([embedding.sin(), embedding.cos()], dim=-1)
        if self.dimensions % 2 == 1:  # pad
            embedding = F.pad(embedding, (0, 1))
        return embedding


class MLP(nn.Module):
    def __init__(
            self,
            in_dimensions: int,
            out_dimensions: int,
            hidden_multiplier: int = 4,
            activation=nn.SiLU()):
        super().__init__()
        hidden_dimensions = in_dimensions * hidden_multiplier
        self.net = nn.Sequential(
            nn.Linear(in_dimensions, hidden_dimensions),
            activation,
            nn.Linear(hidden_dimensions, out_dimensions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock3D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            time_embedding_dimensions=None,
            conditional_dimensions=None,
            groups=8,
            dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embedding_dimensions = time_embedding_dimensions
        self.conditional_dimensions = conditional_dimensions

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        if time_embedding_dimensions is not None:
            self.time_mlp = nn.Linear(time_embedding_dimensions, out_channels)
        else:
            self.time_mlp = None

        if conditional_dimensions is not None:
            self.cond_mlp = nn.Linear(conditional_dimensions, out_channels)
        else:
            self.cond_mlp = None

        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            t_emb=None,
            cond=None) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        if t_emb is not None and self.time_mlp is not None:
            t_add = self.time_mlp(t_emb)[:, :, None, None, None]
            h = h + t_add

        if cond is not None and self.cond_mlp is not None:
            c_add = self.cond_mlp(cond)[:, :, None, None, None]
            h = h + c_add

        h = self.dropout(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return h + self.skip(x)


class CrossAttention(nn.Module):
    """Multi-head attention where keys/values come from condition (CLIP)
    Query: spatial tokens flattened; Key/Value: cond tokens (can be single vector expanded)
    """

    def __init__(
            self,
            dim,
            conditional_dimensions,
            heads=8,
            dim_head=64,
            dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(conditional_dimensions, inner_dim, bias=False)
        self.to_v = nn.Linear(conditional_dimensions, inner_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x: torch.Tensor, cond) -> torch.Tensor:
        # x: (B, N, dim)
        # cond: (B, M, conditional_dimensions)  (M can be 1)
        b, n, _ = x.shape
        m = cond.shape[1]
        h = self.heads

        q = self.to_q(x).view(b, n, h, self.dim_head).transpose(1, 2)  # (B, h, n, dh)
        k = self.to_k(cond).view(b, m, h, self.dim_head).transpose(1, 2)  # (B, h, m, dh)
        v = self.to_v(cond).view(b, m, h, self.dim_head).transpose(1, 2)  # (B, h, m, dh)

        attn = torch.einsum('bhnd,bhmd->bhnm', q, k) * (1.0 / math.sqrt(self.dim_head))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, h * self.dim_head)
        return self.to_out(out)


class SimpleSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, dim)
        b, n, _ = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, n, h, self.dim_head).transpose(1, 2) for t in qkv]
        attn = torch.einsum('bhnd,bhmd->bhnm', q, k) * (1.0 / math.sqrt(self.dim_head))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, h * self.dim_head)
        return self.to_out(out)


class TransformerBottleneck(nn.Module):
    def __init__(self, dim, depth, conditional_dimensions, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                SimpleSelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                nn.LayerNorm(dim),
                CrossAttention(dim, conditional_dimensions, heads=heads, dim_head=dim_head, dropout=dropout),
                nn.LayerNorm(dim),
                nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
            ]))

    def forward(self, x: torch.Tensor, cond) -> torch.Tensor:
        # x: (B, C, r, r, r) -> flatten to tokens
        b, c, r1, r2, r3 = x.shape
        n = r1 * r2 * r3
        x = x.view(b, c, n).permute(0, 2, 1).contiguous()  # (B, N, C)

        for ln1, attn, ln2, cross, ln3, mlp in self.layers:
            x = ln1(x)
            x = attn(x) + x
            x = ln2(x)
            x = cross(x, cond) + x
            x = ln3(x)
            x = mlp(x) + x

        x = x.permute(0, 2, 1).view(b, c, r1, r2, r3)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.op = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.op = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


def center_crop_3d(x, target_shape):
    _, _, d1, d2, d3 = x.shape
    td1, td2, td3 = target_shape
    sd1 = (d1 - td1) // 2
    sd2 = (d2 - td2) // 2
    sd3 = (d3 - td3) // 2
    return x[:, :, sd1:sd1 + td1, sd2:sd2 + td2, sd3:sd3 + td3]


def get_3d_sincos_pos_embed(
        grid_size: int = 16,
        dimensions: int = 30,
        device=None) -> torch.Tensor:
    """
    Generate 3D sinusoidal positional encoding
    Returns: (dim, grid_size, grid_size, grid_size)
    """
    assert dimensions % 6 == 0, f'Positional encoding dim must be divisible by 6, got {dimensions}'

    dim_per_axis = dimensions // 3
    half = dim_per_axis // 2

    emb = torch.zeros(dimensions, grid_size, grid_size, grid_size, device=device)

    # Create frequency bands
    freq = torch.arange(half, device=device).float()
    freq = 1.0 / (10000 ** (freq / float(half)))

    # Create coordinate grids normalized to [-1, 1]
    coords = torch.linspace(-1, 1, grid_size, device=device)
    xs, ys, zs = torch.meshgrid(coords, coords, coords, indexing='ij')

    # Apply sinusoidal encoding for each axis
    for i, grid in enumerate([xs, ys, zs]):
        # grid: (grid_size, grid_size, grid_size)
        # freq: (half,)
        base = torch.einsum('i,jkl->ijkl', freq, grid)  # (half, grid_size, grid_size, grid_size)

        sin = torch.sin(base)
        cos = torch.cos(base)

        emb[i * dim_per_axis: i * dim_per_axis + half] = sin
        emb[i * dim_per_axis + half: i * dim_per_axis + 2 * half] = cos

    return emb
