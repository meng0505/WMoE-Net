







import torch



import torch.nn as nn



import torch.nn.functional as F



from typing import Dict, Optional, List



from ultralytics.nn.modules.conv import Add











def _to_size(y: torch.Tensor, H: int, W: int) -> torch.Tensor:



    """若分辨率不一致则双线性插值到 (H, W)。"""



    if y.shape[-2] != H or y.shape[-1] != W:



        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)



    return y











def _make_norm(norm: str, num_channels: int) -> nn.Module:



    """按名称创建归一化层。"""



    norm = (norm or "bn").lower()



    if norm == "bn":



        return nn.BatchNorm2d(num_channels)



    elif norm == "gn":







        groups = 32



        while groups > 1 and (num_channels % groups != 0):



            groups //= 2



        groups = max(1, groups)



        return nn.GroupNorm(groups, num_channels)



    elif norm == "in":



        return nn.InstanceNorm2d(num_channels, affine=True, track_running_stats=False)



    else:



        raise ValueError(f"Unknown norm: {norm}")















class SingleEnhance(nn.Module):



    """
    轻量单模态增强：1x1 Conv + BN/GN/IN + SiLU + SE，不改变分辨率。
    """



    def __init__(self, c_in: int, c_mid: Optional[int] = None, r: int = 4, norm: str = "bn"):



        super().__init__()



        c_mid = c_mid or c_in



        self.conv = nn.Sequential(



            nn.Conv2d(c_in, c_mid, 1, 1, 0, bias=False),



            _make_norm(norm, c_mid),



            nn.SiLU(inplace=True),



        )



        self.se = nn.Sequential(



            nn.AdaptiveAvgPool2d(1),



            nn.Conv2d(c_mid, max(8, c_mid // r), 1),



            nn.SiLU(inplace=True),



            nn.Conv2d(max(8, c_mid // r), c_mid, 1),



            nn.Sigmoid(),



        )







    def forward(self, x: torch.Tensor) -> torch.Tensor:



        y = self.conv(x)



        y = y * self.se(y)



        return y















class ChannelAlign(nn.Module):



    """
    分组通道对齐：用“对方模态”的全局统计生成本模态的 (γ,β)，温和调制。
    - 分组可防止过拟合；γ 用 tanh 限幅；残差式放大/偏移。
    """



    def __init__(self, c_in: int, groups: int = 8, g_scale: float = 0.5):



        super().__init__()



        assert c_in % groups == 0, "ChannelAlign: c_in 必须能被 groups 整除"



        self.groups = groups



        cg = c_in // groups



        hid = max(16, cg // 2)



        self.mlp = nn.Sequential(



            nn.Linear(cg, hid), nn.ReLU(True),



            nn.Linear(hid, 2 * cg),



        )



        self.g_scale = g_scale







    def forward(self, x: torch.Tensor, cue: torch.Tensor) -> torch.Tensor:



        B, C, H, W = x.shape



        cg = C // self.groups







        s = F.adaptive_avg_pool2d(cue, 1).view(B, self.groups, cg)



        gb = self.mlp(s)



        gamma, beta = gb.split(cg, dim=-1)



        gamma = self.g_scale * torch.tanh(gamma)







        gamma = gamma.reshape(B, C, 1, 1)



        beta  = beta.reshape(B, C, 1, 1)







        return x * (1 + gamma) + beta











class SpatialGate(nn.Module):



    """
    空间门 m(x,y) ∈ [0,1]：告诉“更信任哪一路”。
    用 concat(xr,xs) 过一个轻量卷积栈再接 sigmoid 输出 1 通道门图。
    """



    def __init__(self, c_opt: int, c_sar: int, k: int = 3, norm: str = "bn"):



        super().__init__()



        c_in = c_opt + c_sar



        self.net = nn.Sequential(



            nn.Conv2d(c_in, c_in // 2, k, 1, k // 2, bias=False),



            _make_norm(norm, c_in // 2),



            nn.SiLU(inplace=True),



            nn.Conv2d(c_in // 2, 1, 1, 1, 0, bias=True),



            nn.Sigmoid()



        )







    def forward(self, x_opt: torch.Tensor, x_sar: torch.Tensor) -> torch.Tensor:



        return self.net(torch.cat([x_opt, x_sar], dim=1))











def cosine_corr_map(x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:



    """
    通道维归一化后像素级余弦相似度 → [-1,1]，再映射到 [0,1]
    """



    x1n = F.normalize(x1, p=2, dim=1, eps=eps)



    x2n = F.normalize(x2, p=2, dim=1, eps=eps)



    rho = (x1n * x2n).sum(dim=1, keepdim=True)



    return (rho + 1.0) * 0.5











class MCFM(nn.Module):



    """
    MCFM（Mutually-Guided Consistency-Aware Fusion Module）:
      1) 互引导通道仿射调谐（对方模态 GAP 统计 -> 本模态 gamma/beta）
      2) 空间门控 G（pixel-wise）
      3) 一致性相关图 rho（cosine -> [0,1]）
      4) 一致性加权残差细化
    """



    def __init__(self, c_opt: int, c_sar: int, c_out: int,



                 groups: int = 8, g_scale: float = 0.5, k_gate: int = 3,



                 lambda_refine: float = 0.5, norm: str = "bn"):



        super().__init__()



        self.ca_opt = ChannelAlign(c_opt, groups=groups, g_scale=g_scale)



        self.ca_sar = ChannelAlign(c_sar, groups=groups, g_scale=g_scale)



        self.gate   = SpatialGate(c_opt, c_sar, k=k_gate, norm=norm)



        self.mix    = nn.Sequential(



            nn.Conv2d(c_opt, c_out, 1, 1, 0, bias=False),



            _make_norm(norm, c_out),



            nn.SiLU(inplace=True),



        )







        self.pre_opt = nn.Conv2d(c_opt, c_opt, 1, 1, 0, bias=False)



        self.pre_sar = nn.Conv2d(c_sar, c_opt, 1, 1, 0, bias=False)



        self.lambda_refine = lambda_refine



        self.debug_vis = False



        self.last_debug = None







    def forward(self, x_opt: torch.Tensor, x_sar: torch.Tensor) -> torch.Tensor:







        xo = self.ca_opt(x_opt, cue=x_sar)



        xs = self.ca_sar(x_sar, cue=x_opt)







        xo = self.pre_opt(xo)



        xs = self.pre_sar(xs)







        m = self.gate(xo, xs)







        rho = cosine_corr_map(xo, xs)







        y0 = m * xo + (1.0 - m) * xs



        residual = xo - xs



        y = y0 + self.lambda_refine * rho * residual







        y = self.mix(y)



        y = _to_size(y, x_opt.shape[-2], x_opt.shape[-1])



        if self.debug_vis:



            self.last_debug = {



                "gate": m.detach().cpu(),



                "rho": rho.detach().cpu(),



                "residual_abs": residual.abs().mean(dim=1, keepdim=True).detach().cpu(),



                "lambda_refine": torch.tensor(float(self.lambda_refine), dtype=torch.float32),



            }



        else:



            self.last_debug = None



        return y











class MCFMWrapper(nn.Module):



    """
    仅用 MCFM 做融合（无路由、无先验）
    YAML:
      - [[opt_idx, sar_idx], 1, MCFMWrapper, [c_out, groups, g_scale, k_gate, norm, lambda_refine]]
    """



    def __init__(self, c_out: int, groups: int = 8, g_scale: float = 0.5, k_gate: int = 3, norm: str = "bn",



                 lambda_refine: float = 0.5):



        super().__init__()



        self.c_out = c_out



        self.groups = groups



        self.g_scale = g_scale



        self.k_gate = k_gate



        self.norm = norm



        self.lambda_refine = lambda_refine



        self.block: Optional[MCFM] = None



        self.debug_vis = False







    def forward(self, inputs):



        x_opt, x_sar = inputs



        if self.block is None:



            c_opt, c_sar = x_opt.shape[1], x_sar.shape[1]



            self.block = MCFM(c_opt, c_sar, self.c_out,



                              groups=self.groups, g_scale=self.g_scale, k_gate=self.k_gate,



                              lambda_refine=self.lambda_refine, norm=self.norm).to(x_opt.device)



        self.block.debug_vis = self.debug_vis



        y = self.block(x_opt, x_sar)



        return y















MGCAF = MCFM



MGCAFWrapper = MCFMWrapper











class CrossAdd(nn.Module):



    """Cross expert baseline: y = Conv1x1( (rgb -> match c) (+/avg) sar ) -> c_out"""



    def __init__(self, c_rgb: int, c_sar: int, c_out: int, norm: str = "bn", avg_after_add: bool = False):



        super().__init__()



        self.pre_rgb = nn.Identity() if c_rgb == c_sar else nn.Conv2d(c_rgb, c_sar, 1, bias=False)



        self.add = Add(normalize=avg_after_add)



        self.mix = nn.Sequential(



            nn.Conv2d(c_sar, c_out, 1, bias=False),



            _make_norm(norm, c_out),



            nn.SiLU(inplace=True),



        )







    def forward(self, x_rgb, x_sar):



        xr = self.pre_rgb(x_rgb)



        y  = self.add([xr, x_sar])



        return self.mix(y)















def _grad_energy(x: torch.Tensor) -> torch.Tensor:



    """简单梯度能量作为纹理/清晰度 proxy（通道维度求均值）。"""



    gx = F.pad(x, (1, 1, 1, 1), mode="replicate")



    hx = (gx[..., 1:-1, 2:] - gx[..., 1:-1, :-2]).abs()



    vy = (gx[..., 2:, 1:-1] - gx[..., :-2, 1:-1]).abs()



    e = (hx + vy).mean(dim=(2, 3))



    return e











class FeatureRouter(nn.Module):



    """
    输出 3 路 logits（opt, sar, cross）。
    bias_cross_init 控制融合分支的初始偏置（>0 更容易融合）。
    """



    def __init__(self, c_rgb: int, c_sar: int, hidden: int = 128, T: float = 1.0, bias_cross_init: float = 0.0):



        super().__init__()



        self.T = T



        in_dim = (c_rgb + c_sar) * 3 + 2



        self.gap = nn.AdaptiveAvgPool2d(1)



        self.mlp = nn.Sequential(



            nn.Linear(in_dim, hidden), nn.ReLU(True),



            nn.Linear(hidden, hidden), nn.ReLU(True),



            nn.Linear(hidden, 3),



        )



        with torch.no_grad():



            self.mlp[-1].bias.add_(torch.tensor([0.0, 0.0, bias_cross_init]))







    def forward(self, x_rgb: torch.Tensor, x_sar: torch.Tensor) -> torch.Tensor:







        mr = self.gap(x_rgb).flatten(1)



        ms = self.gap(x_sar).flatten(1)







        vr_raw = x_rgb.var(dim=(2, 3), correction=0)



        vs_raw = x_sar.var(dim=(2, 3), correction=0)



        vr = torch.log1p(torch.clamp_min(vr_raw, 0.0))



        vs = torch.log1p(torch.clamp_min(vs_raw, 0.0))







        gr_raw = _grad_energy(x_rgb)



        gs_raw = _grad_energy(x_sar)



        gr = torch.log1p(torch.clamp_min(gr_raw, 0.0))



        gs = torch.log1p(torch.clamp_min(gs_raw, 0.0))







        q_opt = gr.mean(1, keepdim=True)



        q_sar = gs.mean(1, keepdim=True)







        feat = torch.cat([mr, vr, gr, ms, vs, gs, q_opt - q_sar, q_sar - q_opt], dim=1)



        feat = torch.nan_to_num(feat, nan=0.0, posinf=1e4, neginf=-1e4)



        logits = self.mlp(feat)



        return logits















class PriorBias(nn.Module):



    """
    prior_vec 期望为 [g_opt, g_sar, q_img, p_blur, p_under, p_dense]（缺失可 0 补）
    输出 3 维 logits 偏置，整体乘 alpha 控强度；alpha=0 即“无先验消融”
    """



    def __init__(self, alpha: float = 2.0, in_dim: int = 6):



        super().__init__()



        self.alpha = alpha



        self.mlp = nn.Sequential(



            nn.Linear(in_dim, 16), nn.ReLU(True),



            nn.Linear(16, 3),



        )







    def forward(self, prior_vec: Optional[torch.Tensor]):



        if (prior_vec is None) or (self.alpha <= 0):







            device = self.mlp[0].weight.device



            dtype  = self.mlp[0].weight.dtype



            return torch.zeros((1, 3), device=device, dtype=dtype)



        prior_vec = prior_vec.to(dtype=self.mlp[0].weight.dtype, device=self.mlp[0].weight.device)



        return self.alpha * self.mlp(prior_vec)















class MoEFusion(nn.Module):



    """
    三专家（opt, sar, cross）→ 路由权重 → ConvFusion 融合到 c_out。
    支持：
      - 稠密 softmax（默认）
      - Top-k 稀疏门控（k∈{1,2,3}；建议 1 或 2）
      - 均匀路由（完全不做选择）
      - 仅先验路由（忽略自我统计）
      - STE 硬选软传梯（use_ste=True）
      - Cross 权重地板 cross_floor，防饥饿
      - 可选归一化层类型：norm='bn'|'gn'|'in'
    """



    def __init__(self, c_opt: int, c_sar: int, c_out: int,



                 hidden_router: int = 128, router_T: float = 1.0, prior_alpha: float = 2.0,



                 topk: Optional[int] = None,



                 uniform_routing: bool = False,



                 prior_only: bool = False,



                 bias_cross_init: float = 0.0,



                 use_ste: bool = True,



                 cross_floor: float = 1e-3,



                 mcfm_lambda_refine: float = 0.5,



                 norm: str = "bn"):



        super().__init__()



        self.norm = norm







        self.e_opt = nn.Sequential(



            SingleEnhance(c_opt, c_opt, norm=norm),



            nn.Conv2d(c_opt, c_out, 1, bias=False),



            _make_norm(norm, c_out),



            nn.SiLU(inplace=True),



        )



        self.e_sar = nn.Sequential(



            SingleEnhance(c_sar, c_sar, norm=norm),



            nn.Conv2d(c_sar, c_out, 1, bias=False),



            _make_norm(norm, c_out),



            nn.SiLU(inplace=True),



        )







        self.e_cross = MCFM(c_opt, c_sar, c_out, lambda_refine=mcfm_lambda_refine, norm=norm)







        self.router  = FeatureRouter(c_opt, c_sar, hidden=hidden_router, T=router_T, bias_cross_init=bias_cross_init)



        self.p_bias  = PriorBias(alpha=prior_alpha, in_dim=6)







        self.ipe = nn.Sequential(



            nn.Linear(c_opt + c_sar, max(32, hidden_router // 2)),



            nn.ReLU(True),



            nn.Linear(max(32, hidden_router // 2), 6),



            nn.Sigmoid(),



        )











        self.post_mix = nn.Sequential(



            nn.Conv2d(3 * c_out, c_out, 1, bias=False),



            _make_norm(norm, c_out),



            nn.SiLU(inplace=True),



        )







        self.topk = topk



        self.uniform_routing = uniform_routing



        self.prior_only = prior_only



        self.use_ste = use_ste



        self.cross_floor = float(cross_floor)



        self.last_weights: Optional[torch.Tensor] = None



        self.last_prior_pred: Optional[torch.Tensor] = None



        self.last_prior_target: Optional[torch.Tensor] = None



        self.prior_perturb_mode = "clean"



        self.prior_noise_sigma = 0.0







    def _apply_topk_mask(self, w_soft: torch.Tensor, k: int) -> torch.Tensor:



        """对权重做 Top-k 掩码；k==2 时带配对约束：避免 {opt,sar} 且不含 cross。"""



        B, C = w_soft.shape



        k = max(1, min(k, C))



        vals, idx = torch.topk(w_soft, k, dim=1)



        mask = torch.zeros_like(w_soft).scatter(1, idx, 1.0)











        if k == 2 and C == 3:







            opt_sar = (mask[:, 0].gt(0)) & (mask[:, 1].gt(0)) & mask[:, 2].eq(0)



            if opt_sar.any():



                mask[opt_sar] = 0







                bigger = (w_soft[opt_sar, 0] >= w_soft[opt_sar, 1]).long()



                mask[opt_sar, bigger] = 1.0



                mask[opt_sar, 2] = 1.0



        return mask







    def _apply_prior_perturbation(self, prior_pred: torch.Tensor):



        if self.training:



            return prior_pred, True



        mode = str(getattr(self, "prior_perturb_mode", "clean") or "clean").lower()



        if mode in {"clean", "none", "off"}:



            return prior_pred, True



        if mode in {"dropout", "disable", "no_prior"}:



            return prior_pred, False



        if mode in {"noise", "noisy", "gaussian"}:



            sigma = float(getattr(self, "prior_noise_sigma", 0.0) or 0.0)



            if sigma <= 0:



                return prior_pred, True



            return (prior_pred + torch.randn_like(prior_pred) * sigma).clamp(0.0, 1.0), True



        if mode in {"shuffle", "shuffled"}:



            if prior_pred.shape[0] > 1:



                return prior_pred[torch.randperm(prior_pred.shape[0], device=prior_pred.device)], True



            return prior_pred, True



        raise ValueError(f"Unknown prior_perturb_mode: {mode}")







    def forward(self, x_opt: torch.Tensor, x_sar: torch.Tensor, prior_target: Optional[torch.Tensor] = None):







        y_opt   = self.e_opt(x_opt)



        y_sar   = self.e_sar(x_sar)



        y_cross = self.e_cross(x_opt, x_sar)











        H, W = y_opt.shape[-2], y_opt.shape[-1]



        y_sar   = _to_size(y_sar,   H, W)



        y_cross = _to_size(y_cross, H, W)











        logits  = self.router(x_opt, x_sar)



        if self.prior_only:



            logits = torch.zeros_like(logits)











        s_opt = F.adaptive_avg_pool2d(x_opt, 1).flatten(1)



        s_sar = F.adaptive_avg_pool2d(x_sar, 1).flatten(1)



        prior_pred = self.ipe(torch.cat([s_opt, s_sar], dim=1))



        self.last_prior_pred = prior_pred



        prior_for_bias, use_prior_bias = self._apply_prior_perturbation(prior_pred)











        if self.p_bias.alpha > 0 and use_prior_bias:



            bias = self.p_bias(prior_for_bias)



            Bp = bias.shape[0]



            B  = logits.shape[0]



            if Bp != B:



                if Bp > B:



                    bias = bias[:B]



                else:



                    bias = bias.expand(B, -1)



            logits = logits + bias







        T = max(1e-6, self.router.T)



        w_soft = torch.softmax(logits / T, dim=1)











        if self.uniform_routing:



            w = torch.ones_like(w_soft) / w_soft.size(1)



        elif self.topk is None or self.topk >= w_soft.size(1):







            w = w_soft



        else:







            mask = self._apply_topk_mask(w_soft, self.topk)



            if self.use_ste:







                w = (mask - w_soft).detach() + w_soft



                w = w * mask



            else:



                w = w_soft * mask











            if self.cross_floor > 0 and w.size(1) == 3:



                floor_vec = w.new_tensor([0.0, 0.0, float(self.cross_floor)]).unsqueeze(0)



                w = torch.maximum(w, floor_vec)











            w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)







        self.last_weights = w.detach()



        w = w.view(w.size(0), 3, 1, 1)











        o = w[:, 0:1] * y_opt



        s = w[:, 1:2] * y_sar



        c = w[:, 2:3] * y_cross











        assert o.shape[:1] == s.shape[:1] == c.shape[:1], f"B mismatch: {o.shape}, {s.shape}, {c.shape}"



        assert o.shape[2:] == s.shape[2:] == c.shape[2:], f"HW mismatch: {o.shape}, {s.shape}, {c.shape}"







        y = self.post_mix(torch.cat([o, s, c], dim=1))



        self.last_prior_target = prior_target



        return y, w















class MoEFusionWrapper(nn.Module):



    """
    YAML:
      - [[opt_idx, sar_idx], 1, MoEFusionWrapper,
         [c_out, router_T, prior_alpha, topk, uniform, prior_only, bias_cross,
          use_ste, cross_floor, norm, mcfm_lambda_refine]]
    例：
      [256, 1.0, 2.0, 2, False, False, 0.5, True, 1e-3, 'gn', 0.5]
    """



    def __init__(self, c_out: int, router_T: float = 1.0, prior_alpha: float = 2.0,



                 topk: Optional[int] = None, uniform: bool = False,



                 prior_only: bool = False, bias_cross: float = 0.0,



                 use_ste: bool = True, cross_floor: float = 1e-3, norm: str = "bn",



                 mcfm_lambda_refine: float = 0.5):



        super().__init__()



        self.c_out = c_out



        self.router_T = router_T



        self.prior_alpha = prior_alpha



        self.topk = topk



        self.uniform = uniform



        self.prior_only = prior_only



        self.bias_cross = bias_cross



        self.use_ste = use_ste



        self.cross_floor = cross_floor



        self.norm = norm



        self.mcfm_lambda_refine = mcfm_lambda_refine











        self.block: Optional[MoEFusion] = None











        self.stem2prior: Optional[Dict[str, Dict[str, float]]] = None



        self.no_prior = (prior_alpha <= 0.0)



        self.last_prior_pred: Optional[torch.Tensor] = None



        self.last_prior_target: Optional[torch.Tensor] = None



        self.prior_perturb_mode = "clean"



        self.prior_noise_sigma = 0.0











    def load_gates(self, gates_dict: Dict[str, Dict[str, float]]):



        self.stem2prior = gates_dict







    def _get_prior_vec(self, batch_meta: Optional[Dict]) -> Optional[torch.Tensor]:



        """从 batch meta 取 'stems' → prior_vec（[B,6]）；没有就返回 None。"""







        if (not self.training) or self.no_prior or (self.stem2prior is None):



            return None



        if batch_meta is None:



            return None



        stems: Optional[List[str]] = batch_meta.get("stems", None)



        if stems is None:



            return None







        vecs: List[List[float]] = []



        for s in stems:



            rec = self.stem2prior.get(s, None)



            if rec is None:







                vecs.append([0.5, 0.5, 0.5, 0.0, 0.0, 0.0])



            else:



                vecs.append([



                    float(rec.get("g_opt", 0.5)),



                    float(rec.get("g_sar", 0.5)),



                    float(rec.get("q_img", 0.5)),



                    float(rec.get("p_blur_img", 0.0)),



                    float(rec.get("p_under_img", 0.0)),



                    float(rec.get("p_dense_img", 0.0)),



                ])







        device = next(self.parameters()).device



        dtype  = next(self.parameters()).dtype



        return torch.tensor(vecs, dtype=dtype, device=device)







    def forward(self, inputs, batch_meta: Optional[Dict] = None):



        """
        inputs: [x_rgb, x_sar] 两路特征（已对齐的层，如 C3/C4/C5）
        batch_meta: 可选 dict，需包含 'stems'（图级先验的键）。不传则尝试读 self._current_batch_meta。
        """



        x_opt, x_sar = inputs











        if x_opt.shape[-2:] != x_sar.shape[-2:]:



            x_sar = F.interpolate(x_sar, size=x_opt.shape[-2:], mode="bilinear", align_corners=False)







        if self.block is None:



            c_opt, c_sar = x_opt.shape[1], x_sar.shape[1]



            self.block = MoEFusion(



                c_opt, c_sar, self.c_out,



                router_T=self.router_T, prior_alpha=self.prior_alpha,



                topk=self.topk, uniform_routing=self.uniform,



                prior_only=self.prior_only, bias_cross_init=self.bias_cross,



                use_ste=self.use_ste, cross_floor=self.cross_floor,



                mcfm_lambda_refine=self.mcfm_lambda_refine, norm=self.norm



            ).to(x_opt.device)











        meta = batch_meta or getattr(self, "_current_batch_meta", None)



        prior_target = self._get_prior_vec(meta)











        if prior_target is not None and prior_target.shape[0] != x_opt.shape[0]:



            Bp, B = prior_target.shape[0], x_opt.shape[0]



            if Bp > B:



                prior_target = prior_target[:B]



            else:



                prior_target = prior_target.expand(B, -1)







        self.block.prior_perturb_mode = getattr(self, "prior_perturb_mode", "clean")



        self.block.prior_noise_sigma = float(getattr(self, "prior_noise_sigma", 0.0) or 0.0)







        y, _ = self.block(x_opt, x_sar, prior_target)



        self.last_prior_pred = getattr(self.block, "last_prior_pred", None)



        self.last_prior_target = getattr(self.block, "last_prior_target", None)



        return y







    def prior_supervision_loss(self):



        """SmoothL1(p_hat, p_teacher)。无监督信息时返回 None。"""



        if self.last_prior_pred is None or self.last_prior_target is None:



            return None



        if self.last_prior_pred.shape != self.last_prior_target.shape:



            return None



        return F.smooth_l1_loss(self.last_prior_pred, self.last_prior_target)



