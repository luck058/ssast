# -*- coding: utf-8 -*-
# probing_model.py
# SUPERB-style probing wrapper: frozen encoder + learned weighted sum + linear head.

import torch
import torch.nn as nn


class ProbingModel(nn.Module):
    def __init__(self, frozen_ast_model, task, n_class, f_dim_out, t_dim_out, embed_dim, n_layers=13):
        super().__init__()
        self.encoder = frozen_ast_model
        self.encoder.v.requires_grad_(False)  # freeze entire ViT backbone
        self.task = task
        self.f_dim_out = f_dim_out
        self.t_dim_out = t_dim_out
        self.embed_dim = embed_dim
        self.layer_weights = nn.Parameter(torch.ones(n_layers))  # 13 scalar weights

        if task in ("probe_asr", "probe_pr"):
            self.probe_head = nn.Linear(f_dim_out * embed_dim, n_class)
        elif task == "probe_sid":
            self.probe_head = nn.Linear(embed_dim, n_class)
        else:
            raise ValueError(f"Unknown probe task: {task}")

    def forward(self, x, task=None, **kwargs):
        # x: [B, T, F] — same input shape as ASTModel.forward()
        x = x.unsqueeze(1).transpose(2, 3)  # [B, 1, F, T]
        B = x.shape[0]

        with torch.no_grad():
            all_layers = self.encoder.get_all_intermediate_layers(x)  # list of 13 [B, N, D]

        weights = torch.softmax(self.layer_weights, dim=0)
        weighted = sum(w * h for w, h in zip(weights, all_layers))  # [B, N_patches, D]

        if self.task in ("probe_asr", "probe_pr"):
            # Reshape patches to temporal sequence: [B, t_dim, f_dim*D]
            weighted = weighted.view(B, self.f_dim_out, self.t_dim_out, self.embed_dim)
            weighted = weighted.permute(0, 2, 1, 3).contiguous()  # [B, t, f, D]
            weighted = weighted.view(B, self.t_dim_out, self.f_dim_out * self.embed_dim)
            return self.probe_head(weighted)  # [B, t_dim, n_class]

        elif self.task == "probe_sid":
            weighted = weighted.mean(dim=1)   # [B, D] — mean pool over all patches
            return self.probe_head(weighted)  # [B, n_class]
