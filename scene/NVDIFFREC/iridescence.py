import math
from typing import Tuple

import torch

PI = math.pi

XYZ_TO_RGB = torch.tensor([
    [2.3706743, -0.5138850, 0.0052982],
    [-0.9000405, 1.4253036, -0.0146949],
    [-0.4706338, 0.0885814, 1.0093968],
], dtype=torch.float32)

VAL = torch.tensor([5.4856e-13, 4.4201e-13, 5.2481e-13], dtype=torch.float32)
POS = torch.tensor([1.6810e6, 1.7953e6, 2.2084e6], dtype=torch.float32)
VAR = torch.tensor([4.3278e9, 9.3046e9, 6.6121e9], dtype=torch.float32)
EXTRA_VAL = torch.tensor(9.7470e-14, dtype=torch.float32)
EXTRA_POS = torch.tensor(2.2399e6, dtype=torch.float32)
EXTRA_VAR = torch.tensor(4.5282e9, dtype=torch.float32)
INV_NORMALIZATION = torch.tensor(1.0685e-7, dtype=torch.float32)


def _smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    t = torch.clamp((x - edge0) / (edge1 - edge0), min=0.0, max=1.0)
    return t * t * (3.0 - 2.0 * t)


def _eval_sensitivity(opd: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    phase = 2.0 * PI * opd[..., None] * 1.0e-6
    dtype = opd.dtype
    device = opd.device

    val = VAL.to(device=device, dtype=dtype)
    pos = POS.to(device=device, dtype=dtype)
    var = VAR.to(device=device, dtype=dtype)
    extra_val = EXTRA_VAL.to(device=device, dtype=dtype)
    extra_pos = EXTRA_POS.to(device=device, dtype=dtype)
    extra_var = EXTRA_VAR.to(device=device, dtype=dtype)
    inv_norm = INV_NORMALIZATION.to(device=device, dtype=dtype)

    xyz = val * torch.sqrt(2.0 * PI * var) * torch.cos(pos * phase + shift[..., None]) * torch.exp(-var * phase * phase)
    xyz[..., 0] = xyz[..., 0] + extra_val * torch.sqrt(2.0 * PI * extra_var) * torch.cos(extra_pos * phase[..., 0] + shift[..., 0]) * torch.exp(-extra_var * phase[..., 0] * phase[..., 0])
    return xyz / inv_norm


def _fresnel_coefficients(cos_theta_i: torch.Tensor, n1: torch.Tensor, n2_real: torch.Tensor, n2_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    eps = torch.tensor(1.0e-6, dtype=cos_theta_i.dtype, device=cos_theta_i.device)
    eps_complex = torch.complex(eps, torch.zeros_like(eps))

    cos_theta_i = torch.clamp(cos_theta_i, min=-1.0 + 1.0e-6, max=1.0 - 1.0e-6)
    sin_theta_i_sq = torch.clamp(1.0 - cos_theta_i * cos_theta_i, min=0.0)

    n1_complex = torch.complex(n1, torch.zeros_like(n1))
    n2_complex = torch.complex(n2_real, n2_imag)

    eta = n1_complex / n2_complex
    sin_theta_t_sq = eta * eta * sin_theta_i_sq
    cos_theta_t = torch.sqrt(1.0 - sin_theta_t_sq)

    denom_s = n1_complex * cos_theta_i + n2_complex * cos_theta_t + eps_complex
    denom_p = n2_complex * cos_theta_i + n1_complex * cos_theta_t + eps_complex

    r_s = (n1_complex * cos_theta_i - n2_complex * cos_theta_t) / denom_s
    r_p = (n2_complex * cos_theta_i - n1_complex * cos_theta_t) / denom_p

    R = torch.stack((torch.abs(r_s) ** 2, torch.abs(r_p) ** 2), dim=-1)
    phi = torch.stack((torch.angle(r_s), torch.angle(r_p)), dim=-1)
    return R, phi


def _fresnel_dielectric(cos_theta_i: torch.Tensor, eta2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n1 = torch.ones_like(eta2)
    return _fresnel_coefficients(cos_theta_i, n1, eta2, torch.zeros_like(eta2))


def _fresnel_conductor(cos_theta_i: torch.Tensor, eta2: torch.Tensor, eta3: torch.Tensor, kappa3: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return _fresnel_coefficients(cos_theta_i, eta2, eta3, kappa3)


def compute_iridescent_fresnel(L: torch.Tensor, V: torch.Tensor, N: torch.Tensor, film_thickness: torch.Tensor, eta2: torch.Tensor, eta3: torch.Tensor, kappa3: torch.Tensor) -> torch.Tensor:
    eps = 1.0e-5
    L = torch.nn.functional.normalize(L, dim=-1)
    V = torch.nn.functional.normalize(V, dim=-1)
    N = torch.nn.functional.normalize(N, dim=-1)

    H = torch.nn.functional.normalize(L + V, dim=-1)
    cos_theta1 = torch.clamp(torch.sum(H * L, dim=-1, keepdim=True), min=eps, max=1.0 - eps)

    eta2_eff = torch.lerp(torch.ones_like(eta2), eta2, _smoothstep(0.0, 0.03, film_thickness))
    eta2_eff = torch.clamp(eta2_eff, min=1.0)
    cos_theta2_sq = torch.clamp(1.0 - (1.0 / (eta2_eff * eta2_eff)) * (1.0 - cos_theta1 * cos_theta1), min=0.0)
    cos_theta2 = torch.sqrt(cos_theta2_sq)

    R12, phi12 = _fresnel_dielectric(cos_theta1, eta2_eff)
    R21 = R12
    T121 = torch.clamp(1.0 - R12, min=0.0)
    phi21 = PI - phi12

    R23, phi23 = _fresnel_conductor(cos_theta2, eta2_eff, eta3, kappa3)

    OPD = film_thickness * cos_theta2
    phi2 = phi21 + phi23

    R123 = R12 * R23
    r123 = torch.sqrt(torch.clamp(R123, min=0.0))
    Rs = torch.where(torch.abs(1.0 - R123) > 1.0e-7, (T121 * T121) * R23 / torch.clamp(1.0 - R123, min=1.0e-6), torch.zeros_like(R12))

    C0 = R12 + Rs
    S0 = _eval_sensitivity(torch.zeros_like(OPD), torch.zeros_like(OPD))
    I = 0.5 * (C0[..., 0:1] + C0[..., 1:2]) * S0

    Cm = Rs - T121
    for m in range(1, 4):
        Cm = Cm * r123
        SmS = 2.0 * _eval_sensitivity(m * OPD, m * phi2[..., 0:1])
        SmP = 2.0 * _eval_sensitivity(m * OPD, m * phi2[..., 1:2])
        I = I + 0.5 * (Cm[..., 0:1] * SmS + Cm[..., 1:2] * SmP)

    xyz_to_rgb = XYZ_TO_RGB.to(device=I.device, dtype=I.dtype)
    rgb = torch.matmul(I, xyz_to_rgb.transpose(0, 1))

    ndotl = torch.sum(L * N, dim=-1, keepdim=True)
    ndotv = torch.sum(V * N, dim=-1, keepdim=True)
    mask = ((ndotl > 0.0) & (ndotv > 0.0)).type_as(rgb)

    rgb = torch.nan_to_num(rgb) * mask
    return torch.clamp(rgb, min=0.0, max=1.0)
