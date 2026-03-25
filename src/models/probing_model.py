# -*- coding: utf-8 -*-
# probing_model.py
# SUPERB-style probing wrapper: frozen encoder + learned weighted sum + LSTM/linear head.

import torch
import torch.nn as nn


class ProbingModel(nn.Module):
    def __init__(self, frozen_ast_model, task, n_class, n_layers=13):
        super().__init__()
        self.encoder = frozen_ast_model
        self.encoder.v.requires_grad_(False)   # freeze ViT backbone
        self.encoder.layer_weights.requires_grad_(False)  # not used in probing; freeze to keep param count clean
        self.task = task
        self.t_dim_out = self.encoder.t_dim_out  # exposed for traintest.py
        self.layer_weights = nn.Parameter(torch.ones(n_layers))

        if task in ("probe_asr", "probe_pr"):
            # Freeze the classification head (unused for ASR/PR probing)
            self.encoder.mlp_head.requires_grad_(False)
            # encoder.lstm and encoder.asr_head remain trainable
        elif task in ("probe_cls", "probe_sid"):
            self.encoder.requires_grad_(False)
            self.probe_head = nn.Linear(self.encoder.original_embedding_dim, n_class)
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
            f = self.encoder.f_dim_out
            t = self.encoder.t_dim_out
            D = self.encoder.original_embedding_dim
            # Reshape patches to temporal sequence: [B, t_dim, f_dim*D]
            weighted = weighted.view(B, f, t, D).permute(0, 2, 1, 3).contiguous()
            weighted = weighted.view(B, t, f * D)
            self.encoder.lstm.flatten_parameters()
            weighted, _ = self.encoder.lstm(weighted)      # [B, t_dim, embed_dim*2]
            return self.encoder.asr_head(weighted)         # [B, t_dim, n_class]

        elif self.task in ("probe_cls", "probe_sid"):
            weighted = weighted.mean(dim=1)    # [B, D] — mean pool over all patches
            return self.probe_head(weighted)   # [B, n_class]
