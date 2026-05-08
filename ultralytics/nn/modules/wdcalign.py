







"""
WDCAlign: Wavelet-Directed Cross-Attention Alignment

支持三种模式：
  • HF-only      : enable_hf=True,  use_LL=False
  • LL-only      : enable_hf=False, use_LL=True
  • HF+LL        : enable_hf=True,  use_LL=True

关键点：
  1) LL 分支注意力输出为 d 通道，显式用 toC_LL 将 d→C，再与 Yl_s 残差相加（修复通道不匹配）
  2) 小波边界模式使用 'symmetric'（或将 'reflect' 映射到 'symmetric'）以兼容 PyWavelets
  3) 内部强制 FP32 计算（AMP 友好），输出还原外层 dtype
  4) 老权重兼容：_ensure_compat() & __setstate__() 自动补齐缺失属性/层
"""







import math



import torch



import torch.nn as nn



import torch.nn.functional as F



from pytorch_wavelets import DWTForward, DWTInverse



from torch.cuda.amp import autocast















def window_cross_attention(q, k, v, win, energy_mask=None, rpb=None):



    """
    q, k, v: [B, C, h, w]
    win: odd int (local window size)
    energy_mask: [B,1,h,w] (0=masked out)
    rpb: [win*win] relative position bias (optional)
    return: out [B,C,h,w], attn [B, L, W2]  where L=h*w, W2=win*win
    """



    B, C, h, w = q.shape



    assert win % 2 == 1, f"win must be odd, got {win}"



    L  = h * w



    W2 = win * win



    pad = win // 2



    unfold = nn.Unfold(kernel_size=win, padding=pad)











    q = F.normalize(q, dim=1, eps=1e-6)



    k = F.normalize(k, dim=1, eps=1e-6)











    k_unf = unfold(k).transpose(1, 2).contiguous().view(B, L, W2, C)



    v_unf = unfold(v).transpose(1, 2).contiguous().view(B, L, W2, C)



    q_flat = q.view(B, C, L).transpose(1, 2).contiguous()











    scores = (q_flat.unsqueeze(2) * k_unf).sum(-1) / math.sqrt(C)



    if rpb is not None:



        scores = scores + rpb.view(1, 1, -1).to(scores.dtype)











    scores = scores.clamp(min=-20.0, max=20.0)











    if energy_mask is not None:



        m = unfold(energy_mask).transpose(1, 2).contiguous()



        m = m.to(scores.dtype)



        big_neg = torch.tensor(-1e4, dtype=scores.dtype, device=scores.device)



        scores = scores.masked_fill(m < 0.5, big_neg)











        scores_flat = scores.view(B * L, W2)



        m_flat = m.view(B * L, W2)



        all_masked = (m_flat < 0.5).all(dim=1)



        if all_masked.any():



            center = W2 // 2



            scores_flat[all_masked, :] = big_neg



            scores_flat[all_masked, center] = torch.zeros(1, dtype=scores.dtype, device=scores.device)



        scores = scores_flat.view(B, L, W2)







    attn = F.softmax(scores, dim=-1)



    attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)







    out = (attn.unsqueeze(-1) * v_unf).sum(-2)



    out = out.transpose(1, 2).contiguous().view(B, C, h, w)



    return out, attn















class WDCAlign(nn.Module):



    """
    WDCAlign: Wavelet-Directed Cross-Attention Alignment
      - 小波分解（J=1/2）
      - 在 LH/HL/HH 子带中做局部跨注意力（同方向对子带）
      - iDWT 重建
      - 受控残差注入：f_sar_out = f_sar + λ * ρ * clip(Δ)

    三种模式（通过构造参数选择）：
      • HF-only     : enable_hf=True,  use_LL=False
      • LL-only     : enable_hf=False, use_LL=True
      • HF+LL       : enable_hf=True,  use_LL=True
    """



    def __init__(self,



                 c_in: int,



                 win: int = 7,



                 lambda_scale: float = 0.10,



                 tau_delta: float = 0.5,



                 use_LL: bool = False,



                 gate_k: float = 5.0,



                 gate_t: float = 0.005,



                 energy_thresh: float = 0.0,



                 use_rpb: bool = True,



                 wave: str = 'db1',



                 mode: str = 'symmetric',



                 J: int = 1,



                 dim: int = None,



                 enable_hf: bool = True,



                 use_local_norm: bool = True,



                 local_norm_eps: float = 1e-6



                 ):



        super().__init__()



        self.C = c_in



        self.d = (c_in // 2) if (dim is None) else dim



        self.win = win



        self.lambda_scale = lambda_scale



        self.tau_delta = tau_delta



        self.use_LL = use_LL



        self.enable_hf = enable_hf



        self.energy_thresh = energy_thresh



        self.gate_k, self.gate_t = gate_k, gate_t



        self.use_rpb = use_rpb



        self.J = J



        self.use_local_norm = use_local_norm



        self.local_norm_eps = local_norm_eps



        self.debug_vis = False



        self.last_debug = None











        mode = 'symmetric' if mode == 'reflect' else mode



        self.dwt = DWTForward(J=J, wave=wave, mode=mode)



        self.idwt = DWTInverse(wave=wave, mode=mode)











        def qkv(cin, dim):



            return (nn.Conv2d(cin, dim, 1, bias=True),



                    nn.Conv2d(cin, dim, 1, bias=True),



                    nn.Conv2d(cin, dim, 1, bias=True))



        self.q_LH, self.k_LH, self.v_LH = qkv(self.C, self.d)



        self.q_HL, self.k_HL, self.v_HL = qkv(self.C, self.d)



        self.q_HH, self.k_HH, self.v_HH = qkv(self.C, self.d)











        if use_LL:



            self.q_LL = nn.Conv2d(self.C, self.d, 1, bias=True)



            self.k_LL = nn.Conv2d(self.C, self.d, 1, bias=True)



            self.v_LL = nn.Conv2d(self.C, self.d, 1, bias=True)



            self.ll_lambda = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))



            self.toC_LL = nn.Conv2d(self.d, self.C, 1, bias=True)



            nn.init.kaiming_uniform_(self.toC_LL.weight, a=1.0)



            nn.init.zeros_(self.toC_LL.bias)











        self.toC = nn.Conv2d(self.d, self.C, 1, bias=True)



        nn.init.kaiming_uniform_(self.toC.weight, a=1.0)



        nn.init.zeros_(self.toC.bias)











        self.sub_lambda_LH = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))



        self.sub_lambda_HL = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))



        self.sub_lambda_HH = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))











        sobel_x = torch.tensor([[1, 0, -1],



                                [2, 0, -2],



                                [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)



        sobel_y = torch.tensor([[1, 2, 1],



                                [0, 0, 0],



                                [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)



        self.register_buffer("sobel_x", sobel_x, persistent=False)



        self.register_buffer("sobel_y", sobel_y, persistent=False)











        if use_rpb:



            pad = win // 2



            coords = torch.stack(torch.meshgrid(torch.arange(win), torch.arange(win), indexing='ij'), dim=-1)



            center = torch.tensor([pad, pad]).view(1, 1, 2)



            dist = ((coords - center).float().pow(2).sum(-1)).sqrt()



            rpb = (-dist / (win / 2)).clamp(min=-1).reshape(-1).to(torch.float32)



            self.register_buffer("rpb", rpb, persistent=False)



        else:



            self.register_buffer("rpb", torch.zeros(win * win, dtype=torch.float32), persistent=False)











    def __setstate__(self, state):



        self.__dict__.update(state)







        if not hasattr(self, "enable_hf"):



            self.enable_hf = True



        if not hasattr(self, "use_LL"):



            self.use_LL = False



        if not hasattr(self, "C"):



            cin = None



            for name in ("q_LH", "q_HL", "q_HH"):



                if hasattr(self, name):



                    cin = getattr(self, name).in_channels



                    break



            self.C = cin if cin is not None else 256



        if not hasattr(self, "d"):



            if hasattr(self, "q_LH"):



                self.d = self.q_LH.out_channels



            else:



                self.d = max(32, self.C // 2)



        if self.use_LL and not hasattr(self, "toC_LL"):



            self.toC_LL = nn.Conv2d(self.d, self.C, 1, bias=True)



            nn.init.kaiming_uniform_(self.toC_LL.weight, a=1.0)



            nn.init.zeros_(self.toC_LL.bias)







            if hasattr(self, "q_LL"):



                self.toC_LL = self.toC_LL.to(self.q_LL.weight.device)



        if not hasattr(self, "rpb"):



            win = getattr(self, "win", 7)



            self.register_buffer("rpb", torch.zeros(win * win, dtype=torch.float32), persistent=False)



        if not hasattr(self, "use_local_norm"):



            self.use_local_norm = True



        if not hasattr(self, "local_norm_eps"):



            self.local_norm_eps = 1e-6



        if not hasattr(self, "debug_vis"):



            self.debug_vis = False



        if not hasattr(self, "last_debug"):



            self.last_debug = None











    def _ensure_compat(self):



        if not hasattr(self, "enable_hf"):



            self.enable_hf = True



        if not hasattr(self, "use_LL"):



            self.use_LL = False



        if not hasattr(self, "C"):



            cin = None



            for name in ("q_LH", "q_HL", "q_HH"):



                if hasattr(self, name):



                    cin = getattr(self, name).in_channels



                    break



            self.C = cin if cin is not None else 256



        if not hasattr(self, "d"):



            if hasattr(self, "q_LH"):



                self.d = self.q_LH.out_channels



            else:



                self.d = max(32, self.C // 2)



        if self.use_LL and not hasattr(self, "toC_LL"):



            self.toC_LL = nn.Conv2d(self.d, self.C, 1, bias=True)



            nn.init.kaiming_uniform_(self.toC_LL.weight, a=1.0)



            nn.init.zeros_(self.toC_LL.bias)



            if not hasattr(self, "ll_lambda"):



                self.ll_lambda = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))



        if not hasattr(self, "rpb"):



            win = getattr(self, "win", 7)



            self.register_buffer("rpb", torch.zeros(win * win, dtype=torch.float32), persistent=False)











    def _rho(self, f_opt):







        x = f_opt.mean(1, keepdim=True)



        sobel_x = self.sobel_x.to(dtype=x.dtype, device=x.device)



        sobel_y = self.sobel_y.to(dtype=x.dtype, device=x.device)



        gx = F.conv2d(x, sobel_x, padding=1)



        gy = F.conv2d(x, sobel_y, padding=1)



        g  = torch.sqrt(gx * gx + gy * gy + 1e-12)



        g  = F.avg_pool2d(g, 5, 1, 2)



        rho = torch.sigmoid(self.gate_k * (g - self.gate_t))



        return torch.nan_to_num(rho, nan=0.0)







    def _clip_delta(self, delta):



        """
        L2 向量范数裁剪（可反向传播）：
          若 ||d|| <= tau: 返回 d
          若 ||d|| >  tau: 返回 d * (tau / ||d||)
        """



        B, C, H, W = delta.shape



        v = delta.permute(0, 2, 3, 1).reshape(B * H * W, C)



        n = v.norm(dim=1, keepdim=True).clamp(min=1e-12)



        scale = (self.tau_delta / n).clamp(max=1.0)



        v = v * scale



        v = v.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()



        return torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)







    def _local_magnitude_norm(self, delta):



        """3x3 局部幅度归一化，抑制散射峰值导致的过注入。"""



        mag = F.avg_pool2d(delta.abs(), kernel_size=3, stride=1, padding=1)



        mag = mag + self.local_norm_eps



        return delta / mag







    def _energy_mask(self, LH, HL, HH):







        e = (LH.abs() + HL.abs() + HH.abs()).mean(dim=1, keepdim=True)



        return (e > self.energy_thresh).float()







    @staticmethod



    def _band_energy(x):



        """Per-image mean absolute energy [B]."""



        return x.abs().mean(dim=(1, 2, 3))











    def forward(self, x, f_sar=None):



        """
        兼容两种调用：
          1) x 为 [f_opt, f_sar]
          2) forward(f_opt, f_sar)
        返回：f_sar_out: [B,C,H,W]（与输入相同 dtype）
        """







        self._ensure_compat()











        if f_sar is None:
            assert isinstance(x, (list, tuple)) and len(x) == 2, f"WDCAlign expects [opt, sar], got {type(x)}"



            f_opt, f_sar = x



        else:



            f_opt = x







        orig_dtype = f_opt.dtype











        with autocast(enabled=False):







            if next(self.parameters()).dtype != torch.float32:



                self.float()



            self.dwt.float()



            self.idwt.float()











            f_opt32 = f_opt.to(torch.float32)



            f_sar32 = f_sar.to(torch.float32)











            Yl_o, Yh_o = self.dwt(f_opt32)



            Yl_s, Yh_s = self.dwt(f_sar32)











            idx = 0 if self.J == 1 else 1



            LH_o, HL_o, HH_o = Yh_o[idx][:, :, 0], Yh_o[idx][:, :, 1], Yh_o[idx][:, :, 2]



            LH_s, HL_s, HH_s = Yh_s[idx][:, :, 0], Yh_s[idx][:, :, 1], Yh_s[idx][:, :, 2]











            mask = None if self.energy_thresh <= 0 else self._energy_mask(LH_s, HL_s, HH_s)











            if getattr(self, "enable_hf", True):



                q = self.q_LH(LH_s); k = self.k_LH(LH_o); v = self.v_LH(LH_o)



                Y_LH, _ = window_cross_attention(q, k, v, self.win, energy_mask=mask, rpb=self.rpb)







                q = self.q_HL(HL_s); k = self.k_HL(HL_o); v = self.v_HL(HL_o)



                Y_HL, _ = window_cross_attention(q, k, v, self.win, energy_mask=mask, rpb=self.rpb)







                q = self.q_HH(HH_s); k = self.k_HH(HH_o); v = self.v_HH(HH_o)



                Y_HH, _ = window_cross_attention(q, k, v, self.win, energy_mask=mask, rpb=self.rpb)











                LH_pred = self.toC(Y_LH)



                HL_pred = self.toC(Y_HL)



                HH_pred = self.toC(Y_HH)







                Y_LH_c = LH_s + torch.tanh(self.sub_lambda_LH) * (LH_pred - LH_s)



                Y_HL_c = HL_s + torch.tanh(self.sub_lambda_HL) * (HL_pred - HL_s)



                Y_HH_c = HH_s + torch.tanh(self.sub_lambda_HH) * (HH_pred - HH_s)



            else:







                Y_LH_c, Y_HL_c, Y_HH_c = LH_s, HL_s, HH_s







            Yh_s_aligned = list(Yh_s)



            Yh_s_aligned[idx] = torch.stack([Y_LH_c, Y_HL_c, Y_HH_c], dim=2)











            LLs_aligned = Yl_s



            if getattr(self, "use_LL", False):



                q = self.q_LL(Yl_s); k = self.k_LL(Yl_o); v = self.v_LL(Yl_o)



                Y_LL, _ = window_cross_attention(q, k, v, self.win, energy_mask=None, rpb=self.rpb)







                Y_LL = self.toC_LL(Y_LL)







                LLs_aligned = Yl_s + torch.tanh(self.ll_lambda) * (Y_LL - Yl_s)











            f_tilde32 = self.idwt((LLs_aligned, Yh_s_aligned))



            Delta32 = f_tilde32 - f_sar32



            if self.use_local_norm:



                Delta32 = self._local_magnitude_norm(Delta32)



            Delta32 = self._clip_delta(Delta32)



            rho32   = self._rho(f_opt32)



            out32   = f_sar32 + self.lambda_scale * rho32 * Delta32











            if self.debug_vis:



                hf_before = self._band_energy(LH_s) + self._band_energy(HL_s) + self._band_energy(HH_s)



                hf_after  = self._band_energy(Y_LH_c) + self._band_energy(Y_HL_c) + self._band_energy(Y_HH_c)



                ll_before = self._band_energy(Yl_s)



                ll_after  = self._band_energy(LLs_aligned)



                self.last_debug = {



                    "rho": rho32.detach().cpu(),



                    "delta_abs": Delta32.abs().mean(dim=1, keepdim=True).detach().cpu(),



                    "hf_before": hf_before.detach().cpu(),



                    "hf_after": hf_after.detach().cpu(),



                    "ll_before": ll_before.detach().cpu(),



                    "ll_after": ll_after.detach().cpu(),



                    "sub_lambda": torch.tensor(



                        [



                            float(torch.tanh(self.sub_lambda_LH).item()),



                            float(torch.tanh(self.sub_lambda_HL).item()),



                            float(torch.tanh(self.sub_lambda_HH).item()),



                        ],



                        dtype=torch.float32,



                    ),



                    "ll_lambda": torch.tensor(



                        float(torch.tanh(self.ll_lambda).item()) if getattr(self, "use_LL", False) else 0.0,



                        dtype=torch.float32,



                    ),



                    "lambda_scale": torch.tensor(float(self.lambda_scale), dtype=torch.float32),



                }



            else:



                self.last_debug = None











        return out32.to(orig_dtype)



