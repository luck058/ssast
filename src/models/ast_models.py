# -*- coding: utf-8 -*-
# @Time    : 7/16/21 3:12 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ast_models.py

# the unified ast models for all pretraining/fine-tuning tasks.

import torch.nn as nn
import torch
import sys
sys.path.append("/data/sls/scratch/yuangong/aed-trans/src/models/")
sys.path.append("/data/sls/scratch/yuangong/aed-trans/src/")
import timm
import numpy as np
from timm.models.layers import to_2tuple
from random import randrange
from matplotlib import pyplot as plt
import random
from torch.nn import functional as F

try:
    # timm >= 0.9.x moves layers to timm.layers
    from timm.layers import trunc_normal_, PatchEmbed
except ImportError:
    # timm < 0.9.x keeps them in timm.models.layers
    from timm.models.layers import trunc_normal_
    try:
        from timm.models.layers import PatchEmbed
    except ImportError:
        # Fallback for very old timm versions
        from timm.models.vision_transformer import PatchEmbed
# -----------------------------------------------

# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, **kwargs):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2) # BCHW -> BNC
        x = self.norm(x)
        return x

def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class ASTModel(nn.Module):
    def __init__(self, label_dim=527,
                 fshape=128, tshape=2, fstride=128, tstride=2,
                 input_fdim=128, input_tdim=1024, model_size='base',
                 pretrain_stage=True, load_pretrained_mdl_path=None,
                 num_clusters=512, target_layer_idx=6, vocab_size=30):

        super(ASTModel, self).__init__()
        # override timm input shape restriction
        if hasattr(timm.models.vision_transformer, 'PatchEmbed'):
            timm.models.vision_transformer.PatchEmbed = PatchEmbed
        
        # Update timm.layers if it exists (timm >= 0.9.x)
        if hasattr(timm, 'layers') and hasattr(timm.layers, 'PatchEmbed'):
            timm.layers.PatchEmbed = PatchEmbed
        # -----------------------------------------------

        # pretrain the AST models
        if pretrain_stage == True:
            if load_pretrained_mdl_path != None:
                raise ValueError('Setting load_pretrained_mdl_path at pretraining stage is useless, pretraining is always from scratch, please change it to None.')
            if fstride != fshape or tstride != tshape:
                raise ValueError('fstride != fshape or tstride != tshape, they must be same at the pretraining stage, patch split overlapping is not supported.')

            # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
            if model_size == 'tiny':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=False)
                self.heads, self.depth = 3, 12
                self.cls_token_num = 2
            elif model_size == 'small':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=False)
                self.heads, self.depth = 6, 12
                self.cls_token_num = 2
            elif model_size == 'base':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=False)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 2
            elif model_size == 'base_nokd':
                self.v = timm.create_model('vit_deit_base_patch16_384', pretrained=False)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 1
            else:
                raise Exception('Model size must be one of tiny, small, base, base_nokd')

            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            # SSL Pretraining Code
            self.softmax = nn.Softmax(dim=-1)
            self.lsoftmax = nn.LogSoftmax(dim=-1)
            self.fshape, self.tshape = fshape, tshape
            self.fstride, self.tstride = fstride, tstride
            self.input_fdim, self.input_tdim = input_fdim, input_tdim
            # this is a trick to make state_dict to track pretraining input_fdim and input_tdim and save them by using torch.save
            self.p_input_fdim, self.p_input_tdim = nn.Parameter(torch.tensor(input_fdim), requires_grad=False), nn.Parameter(torch.tensor(input_tdim), requires_grad=False)

            # masked patch classification (discriminative objective) layer
            # we use two layers for pretext task, but using a single layer has similar performance.
            # we map the output of transformer (768-dim for base models) to 256-dim patch input space, and then dot product with flattened patch input (also 256-dim) to calculate loss.
            # alternatively, you can map the output of transformer to 768-dim patch embedding space, and dot product with patch embedding. Performance-wise they are similar, but map to 256 space is more efficient.
            self.cpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
            # masked patch reconstruction (generative objective) layer
            self.gpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
            self.unfold = torch.nn.Unfold(kernel_size=(fshape, tshape), stride=(fstride, tstride))

            # we use learnable mask embedding (follow the BEIT paper), but using a fixed mask embedding (e.g., 0) leads to same performance.
            self.mask_embed = nn.Parameter(torch.zeros([1, 1, self.original_embedding_dim]))
            self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)

            # get the intermediate shape
            self.p_f_dim, self.p_t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            num_patches = self.p_f_dim * self.p_t_dim
            self.num_patches = num_patches
            self.v.patch_embed.num_patches = num_patches
            print('pretraining patch split stride: frequency={:d}, time={:d}'.format(fstride, tstride))
            print('pretraining patch shape: frequency={:d}, time={:d}'.format(fshape, tshape))
            print('pretraining patch array dimension: frequency={:d}, time={:d}'.format(self.p_f_dim, self.p_t_dim))
            print('pretraining number of patches={:d}'.format(num_patches))

            # the linear patch projection layer, use 1 channel for spectrogram rather than the original 3 channels for RGB images.
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
            self.v.patch_embed.proj = new_proj

            # use trainable positional embedding
            new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + self.cls_token_num, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)
            
            # MelHuBERT loss initialisation
            self.mhb_pred_layer = nn.Linear(self.original_embedding_dim, num_clusters)
            self.patch_dim = fshape * tshape 
            self.num_clusters = num_clusters
            self.target_layer_idx = target_layer_idx
            
            # Determine dimension for centroids
            # If using raw patch features
            if self.target_layer_idx == -1:
                # Use raw patch dimension (e.g., 16*16 = 256)
                self.centroid_dim = self.patch_dim
            else:
                # Use Transformer embedding dimension (e.g., 768)
                self.centroid_dim = self.original_embedding_dim

            # Buffer for centroids(Not a Parameter, won't be updated by optimizer)
            self.register_buffer('cluster_centroids', torch.randn(self.num_clusters, self.centroid_dim))
            self.cluster_centroids.data = F.normalize(self.cluster_centroids.data, p=2, dim=1)
            

        # use a pretrained models for finetuning
        elif pretrain_stage == False:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if load_pretrained_mdl_path == None:
                raise ValueError('Please set load_pretrained_mdl_path to load a pretrained models.')
            sd = torch.load(load_pretrained_mdl_path, map_location=device)
            # Remove cluster centroids as they are unused to avoid shape mismatch
            if 'module.cluster_centroids' in sd:
                print("Removing unused cluster_centroids from checkpoint to avoid shape mismatch.")
                del sd['module.cluster_centroids']
            if 'cluster_centroids' in sd:
                del sd['cluster_centroids']
            # get the fshape and tshape, input_fdim and input_tdim in the pretraining stage
            try:
                p_fshape, p_tshape = sd['module.v.patch_embed.proj.weight'].shape[2], sd['module.v.patch_embed.proj.weight'].shape[3]
                p_input_fdim, p_input_tdim = sd['module.p_input_fdim'].item(), sd['module.p_input_tdim'].item()
            except:
                raise  ValueError('The model loaded is not from a torch.nn.Dataparallel object. Wrap it with torch.nn.Dataparallel and try again.')

            print('now load a SSL pretrained models from ' + load_pretrained_mdl_path)
            # during pretraining, fstride=fshape and tstride=tshape because no patch overlapping is used
            # here, input_fdim and input_tdim should be that used in pretraining, not that in the fine-tuning.
            # we need to know input_fdim and input_tdim to do positional embedding cut/interpolation.
            # generally it should be better to use same input_fdim during pretraining and finetuning, but input_tdim can be safely different
            audio_model = ASTModel(fstride=p_fshape, tstride=p_tshape, fshape=p_fshape, tshape=p_tshape,
                                   input_fdim=p_input_fdim, input_tdim=p_input_tdim, pretrain_stage=True, model_size=model_size,
                                   num_clusters=num_clusters)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)

            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.cls_token_num = audio_model.module.cls_token_num

            # mlp head for fine-tuning
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim),
                                          nn.Linear(self.original_embedding_dim, label_dim))
            
            # ASR head for fine-tuning
            self.f_dim_out = (input_fdim - fshape) // fstride + 1
            self.t_dim_out = (input_tdim - tshape) // tstride + 1
            lstm_input_dim = self.original_embedding_dim * self.f_dim_out
            
            # Tiny: 1 layer = ~0.6M params in LSTM layer
            # Tiny: 2 layers = ~1.8M params in LSTM layers
            # Base: 2 layer = ~20M params in LSTM layers
            if model_size == 'tiny':
                lstm_layers = 1
            else:
                lstm_layers = 2
            
            self.lstm = nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=self.original_embedding_dim,  # 192 for tiny, 768 for base
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True
            )
            self.asr_head = nn.Linear(self.original_embedding_dim * 2, vocab_size)

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            # patch array dimension during pretraining
            p_f_dim, p_t_dim = audio_model.module.p_f_dim, audio_model.module.p_t_dim
            num_patches = f_dim * t_dim
            p_num_patches = p_f_dim * p_t_dim
            self.v.patch_embed.num_patches = num_patches
            print('fine-tuning patch split stride: frequncey={:d}, time={:d}'.format(fstride, tstride))
            print('fine-tuning number of patches={:d}'.format(num_patches))

            # patch shape should be same for pretraining and fine-tuning
            if fshape != p_fshape or tshape != p_tshape:
                raise ValueError('The patch shape of pretraining and fine-tuning is not consistant, pretraining: f={:d}, t={:d}, finetuning: f={:d}, t={:d}'.format(p_fshape, p_tshape, fshape, tshape))

            # patch split stride generally should be different for pretraining and fine-tuning, as patch split overlapping is only used in finetuning
            # during pretraining, p_fshape = p_fstride and p_tshape = p_tstride
            if fstride != p_fshape or tstride != p_tshape:
                # initialize a new patch embedding layer with desired new stride.
                new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
                # but the weights of patch embedding layer is still got from the pretrained models
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
                self.v.patch_embed.proj = new_proj

            new_pos_embed = self.v.pos_embed[:, self.cls_token_num:, :].detach().reshape(1, p_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, p_f_dim, p_t_dim)
            # cut or interpolate the positional embedding
            if t_dim < p_t_dim:
                new_pos_embed = new_pos_embed[:, :, :, int(p_t_dim/2) - int(t_dim / 2): int(p_t_dim/2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(p_f_dim, t_dim), mode='bilinear')
            if f_dim < p_f_dim:
                new_pos_embed = new_pos_embed[:, :, int(p_f_dim/2) - int(f_dim / 2): int(p_f_dim/2) - int(f_dim / 2) + f_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')

            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :self.cls_token_num, :].detach(), new_pos_embed], dim=1))

    # get the shape of intermediate representation.
    def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    # generate mask for 16*16 patch
    def gen_maskid_patch(self, sequence_len=512, mask_size=100, cluster=3):
        mask_id = []

        # randomize clutering factor in [3,6)
        cur_clus = randrange(cluster) + 3

        while len(list(set(mask_id))) <= mask_size:
            start_id = randrange(sequence_len)

            # this improves the efficiency, but might change the pretrained model
            # while start_id in mask_id:
            #     start_id = randrange(sequence_len)

            cur_mask = []
            for i in range(0, cur_clus):
                for j in range(0, cur_clus):
                    mask_cand = start_id + self.p_t_dim * i + j
                    if mask_cand > 0 and mask_cand < sequence_len:
                        cur_mask.append(mask_cand)
            mask_id = mask_id + cur_mask
        mask_id = list(set(mask_id))[:mask_size]
        return torch.tensor(mask_id)

    # using cluster for frame masking hurts the performance, so just use the naive random sampling
    def gen_maskid_frame(self, sequence_len=512, mask_size=100):
        mask_id = random.sample(range(0, sequence_len), mask_size)
        return torch.tensor(mask_id)

    def finetuningavgtok(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)

        # average output of all tokens except cls token(s)
        x = torch.mean(x[:, self.cls_token_num:, :], dim=1)
        x = self.mlp_head(x)
        return x

    def finetuningcls(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)

        # if models has two cls tokens (DEIT), average as the clip-level representation
        if self.cls_token_num == 2:
            x = (x[:, 0] + x[:, 1]) / 2
        else:
            x = x[:, 0]
        x = self.mlp_head(x)
        return x

    def finetuningasr(self, x):
        x = self.v.patch_embed(x)
        B, N, D = x.shape
        
        # Add tokens & Pos Embed
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        for blk in self.v.blocks: x = blk(x)
        x = self.v.norm(x)
        
        # Reshape for LSTM
        x = x[:, self.cls_token_num:, :] # Remove CLS
        x = x.transpose(1, 2).view(B, D, self.f_dim_out, self.t_dim_out)
        x = x.permute(0, 3, 2, 1).contiguous() # [Batch, Time, Freq, Dim]
        x = x.view(B, self.t_dim_out, self.f_dim_out * D)
        
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return self.asr_head(x)
        

    def get_intermediate_layers(self, x, layer_idx):
        """
        Extract Unmasked Features from Transformer Layer 'layer_idx'
        If layer_idx is -1, returns the RAW Unfolded Patches (Spectrogram chunks).
        If layer_idx is >= 0, returns the output of that Transformer Block.
        """
        if layer_idx == -1:
            # x is [Batch, 1, F, T] (e.g., [B, 1, 128, 1024])
            # self.unfold was defined in __init__ as:
            # torch.nn.Unfold(kernel_size=(fshape, tshape), stride=(fstride, tstride))
            
            # Extract patches: Output is [B, Patch_Dim, Num_Patches]
            patches = self.unfold(x)
            
            # Transpose to [B, Num_Patches, Patch_Dim] to match Transformer output shape
            patches = patches.transpose(1, 2)
            
            return patches
        else:
            # Pass through patch embedding
            x = self.v.patch_embed(x)
            B = x.shape[0]
            # Add cls and dist tokens
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            if self.cls_token_num == 2:
                dist_token = self.v.dist_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, dist_token, x), dim=1)
            else:
                x = torch.cat((cls_tokens, x), dim=1)
            # Add positional embeddings
            x = x + self.v.pos_embed
            # Apply dropout
            x = self.v.pos_drop(x)
            # Pass through Transformer blocks up to layer_idx
            for i, blk in enumerate(self.v.blocks):
                x = blk(x)
                if i == layer_idx:
                    break
            return x[:, self.cls_token_num:, :]

    @torch.no_grad()
    def get_cluster_labels(self, x):
        """Helper for Dataloader Labeling"""
        # Run pass to get intermediate features
        # Returns [B, N, 256] if target_layer_idx == -1
        target_features = self.get_intermediate_layers(x, self.target_layer_idx)
        # Match to centroids
        flat_features = target_features.contiguous().view(-1, self.centroid_dim)
        flat_features = F.normalize(flat_features, p=2, dim=1)
        centroids = F.normalize(self.cluster_centroids, p=2, dim=1)
        # Cosine Similarity -> Argmax
        similarity = torch.matmul(flat_features, centroids.transpose(0, 1))
        flat_target_ids = torch.argmax(similarity, dim=1)
        # Reshape [B, N_patches] and return
        return flat_target_ids.view(x.shape[0], -1).cpu()
    
    # General model body: handles patch embedding, masking, and transformer encoding
    def _masked_encoding_body(self, x, mask_patch, cluster):
        # Unfold input to get raw patches (ground truth targets)
        # x shape: (batch_size, sequence_len, embedding dim)
        input_patches = self.unfold(x).transpose(1, 2)
        B = x.shape[0]

        # Patch embedding
        x = self.v.patch_embed(x)

        # Initialize mask index and dense mask
        # mask_index shape: B * mask_patch
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        # mask_dense shape: B * sequence_len * hidden_dim
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)

        # Generate masks for each audio clip in the batch
        for i in range(B):
            # Randomly generate #mask_patch mask indexes without duplicate
            if cluster == True:
                # Use this if you are masking e.g. 16*16 patches
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                # Use this if you are masking frame, i.e., 128*2 patches
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            
            # Mask the dense tensor (set masked areas to 0)
            mask_dense[i, mask_index[i], :] = 0

        # Apply mask tokens
        # Follow BEIT paper, mask with learnable masking embedding
        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)
        x = x * mask_dense + (1 - mask_dense) * mask_tokens

        # Pass through the Transformer layers
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        if self.cls_token_num == 2:
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)

        # Return encoded features, raw input patches (targets), and mask indices
        return x, input_patches, mask_index
    
    def _mpc_head(self, x, input_patches, mask_index, mask_patch, show_mask=False):
        B = x.shape[0]

        # Prepare the true values of masked samples (ground truth)
        encode_samples = torch.empty((B, mask_patch, 256), device=x.device, requires_grad=False).float()
        for i in range(B):
            encode_samples[i] = input_patches[i, mask_index[i], :].clone().detach()

        # Prediction of the masked patch
        pred = torch.empty((B, mask_patch, 256), device=x.device).float()
        for i in range(B):
            # Map output of transformer to patch input space
            # + self.cls_token_num to skip cls and dist tokens
            pred[i] = self.cpredlayer(x[i, mask_index[i] + self.cls_token_num, :])

        # Calculate the NCE loss
        nce = torch.tensor(0.0).to(x.device)
        correct = torch.tensor(0.0).to(x.device)
        for i in np.arange(0, B):
            # Negative samples are from the same batch
            # NOTE 8/12/2022: has a difference with equation (1) in the ssast paper but (likely) performance-wise similar, see https://github.com/YuanGongND/ssast/issues/13
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            correct += torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, mask_patch, device=x.device)))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        
        acc = 1. * correct / (B * mask_patch)
        nce = nce / (-1. * B * mask_patch)

        # Visualize the masked area (probing test only)
        if show_mask == False:
            return acc, nce
        else:
            if B > 1:
                raise Exception('Currently only support single spectrogram probing test.')

            self.mask_correct = torch.nn.Parameter(torch.arange(0, mask_patch), requires_grad=False)
            
            # Visualization relies on the raw patches (input_patches)
            pred_vis = input_patches.clone()
            masked_vis = input_patches.clone()

            for i in range(B):
                result = [float(t) * 99 for t in torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct)]
                pred_vis[i, mask_index[i], :] = torch.tensor(result).reshape(mask_patch, 1).expand(mask_patch, 256)
                masked_vis[i, mask_index[i], :] = 99.0

            fold = torch.nn.Fold(output_size=([self.input_fdim, self.input_tdim]), kernel_size=(self.fshape, self.tshape), stride=(self.fstride, self.tstride))
            pred_vis = fold(pred_vis.transpose(1, 2))
            masked_vis = fold(masked_vis.transpose(1, 2))

            return pred_vis, masked_vis

    def _mpg_head(self, x, input_patches, mask_index, mask_patch):
        B = x.shape[0]
        pred = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float()
        target = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float()

        for i in range(B):
            # + self.cls_token_num to skip cls and dist tokens
            # Project transformer output to reconstruction dimension
            pred[i] = self.gpredlayer(x[i, mask_index[i] + self.cls_token_num, :])
            # Target is the raw input patch
            target[i] = input_patches[i, mask_index[i], :]
        
        mse = torch.mean((pred - target) ** 2)
        return mse

    def _mpmhb_head(self, x, target_ids, mask_index, mask_patch):
        """
        Calculates the Cross Entropy loss between predicted logits and target cluster IDs.
        """
        B = x.shape[0]
        
        # Select the masked tokens from the transformer output (Student Prediction)
        masked_output = torch.empty((B, mask_patch, self.original_embedding_dim), device=x.device)
        # Select the corresponding target IDs (Ground Truth)
        batch_target_ids = torch.empty((B, mask_patch), device=x.device, dtype=torch.long)
        
        for i in range(B):
            masked_output[i] = x[i, mask_index[i] + self.cls_token_num, :]
            batch_target_ids[i] = target_ids[i, mask_index[i]]

        # Predict Cluster Logits
        logits = self.mhb_pred_layer(masked_output)
        
        # Calculate Loss
        return F.cross_entropy(logits.view(-1, self.num_clusters), batch_target_ids.view(-1))
    
    # Masked patch pretraining with discriminative objective
    def mpc(self, x, mask_patch, cluster, show_mask=False):
        """Masked patch pretraining with discriminative objective"""
        # General Model Body
        x, input_patches, mask_index = self._masked_encoding_body(x, mask_patch, cluster)
        # MPC Head Logic
        # If show_mask is True, return the visualization of masked area instead of loss/accuracy
        acc, nce = self._mpc_head(x, input_patches, mask_index, mask_patch, show_mask)
        return nce, acc
        
    # Masked patch pretraining with generative objective
    def mpg(self, x, mask_patch, cluster):
        """Masked patch pretraining with generative objective"""
        # General Model Body
        x, input_patches, mask_index = self._masked_encoding_body(x, mask_patch, cluster)
        # MPG Head Logic
        mse = self._mpg_head(x, input_patches, mask_index, mask_patch)

        return mse
    
    # Masked patch joint pretraining with generative and discriminative objective
    def mpj(self, x, mask_patch, cluster, mpg_weight=10):
        """Masked patch joint pretraining with generative and discriminative objective. Loss = mpc_loss + mpg_weight * mpg_loss
        """
        # General Model Body
        x, input_patches, mask_index = self._masked_encoding_body(x, mask_patch, cluster)
        # Both MPC and MPG Head Logic
        mse = self._mpg_head(x, input_patches, mask_index, mask_patch)
        acc, nce = self._mpc_head(x, input_patches, mask_index, mask_patch, show_mask=False)
        combined_loss = nce + (mpg_weight * mse)
        return combined_loss, acc
    
    def mpmhb(self, x, mask_patch, cluster, target_ids=None, args=None):
        """Masked patch joint pretraining with MelHuBERT objective, discriminative and generative objective."""
        if args is None:
            raise ValueError("args must be provided for mpmhb to specify mpg_weight and mpmhb_weight")
        if args['mpg_weight'] == 0 and args['mpmhb_weight'] == 0 and args['mpc_weight'] == 0:
            raise ValueError("At least one of mpg_weight, mhb_weight, or mpc_weight must be non-zero")
        
        # General Model Body
        x_masked, input_patches, mask_index = self._masked_encoding_body(x, mask_patch, cluster)
        
        # Run MPC loss
        if args["mpc_weight"] != 0:
            acc_mpc, loss_mpc = self._mpc_head(x_masked, input_patches, mask_index, mask_patch, show_mask=False)
        else:
            acc_mpc = torch.tensor(0.0, device=x.device)
            loss_mpc = torch.tensor(0.0, device=x.device)
            
        # Run MPG loss
        if args["mpg_weight"] != 0:
            loss_mpg = self._mpg_head(x_masked, input_patches, mask_index, mask_patch)
        else:
            loss_mpg = torch.tensor(0.0, device=x.device)
        
        # Run MPMHB loss
        if target_ids is None and args["mpmhb_weight"] != 0:
            raise ValueError("target_ids must be provided for mpmhb")
        if args["mpmhb_weight"] != 0:
            loss_mpmhb = self._mpmhb_head(x_masked, target_ids, mask_index, mask_patch)
        else:
            loss_mpmhb = torch.tensor(0.0, device=x.device)
        
        # Weighted sum of losses
        total_loss = (args['mpg_weight'] * loss_mpg) + (args['mpc_weight'] * loss_mpc) + (args['mpmhb_weight'] * loss_mpmhb)
        
        return total_loss, acc_mpc, loss_mpg, loss_mpc, loss_mpmhb
    
    def forward(self, x, task, cluster=True, mask_patch=400, target_ids=None, args=None):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        # finetuning (ft), use the mean of all token (patch) output as clip-level representation.
        # this is default for SSAST fine-tuning as during pretraining, supervision signal is given to each token, not the [cls] token
        if task == 'ft_avgtok':
            return self.finetuningavgtok(x)
        # alternatively, use the [cls] token output as clip-level representation.
        elif task == 'ft_cls':
            return self.finetuningcls(x)
        elif task == 'ft_asr':
            return self.finetuningasr(x)
        # pretraining, masked patch classification (discriminative objective)
        elif task == 'pretrain_mpc':
            return self.mpc(x, mask_patch=mask_patch, cluster=cluster)
        # pretraining, masked patch reconstruction (generative objective)
        elif task == 'pretrain_mpg':
            return self.mpg(x, mask_patch=mask_patch, cluster=cluster)
        elif task == 'pretrain_mpj':
            if args is not None and 'mpg_weight' in args:
                mpg_weight = args['mpg_weight']
            else:
                mpg_weight = 10
            return self.mpj(x, mask_patch=mask_patch, cluster=cluster, mpg_weight=mpg_weight)
        elif task == 'pretrain_mpmhb':
            return self.mpmhb(x, mask_patch=mask_patch, cluster=cluster, target_ids=target_ids, args=args)
        elif task == 'visualize_mask':
            return self.mpc(x, mask_patch=mask_patch, cluster=cluster, show_mask=True)
        else:
            raise Exception('Task unrecognized.')

if __name__ == '__main__':
    # this is an example of how to use the SSAST model

    # pretraining stage
    # suppose you have an unlabled dataset with avg length of 1024 frames (i.e., 10.24s)
    input_tdim = 1024
    # create a 16*16 patch based AST model for pretraining.
    # note, we don't use patch split overlap in pretraining, so fstride=fshape and tstride=tshape
    ast_mdl = ASTModel(
                 fshape=16, tshape=16, fstride=16, tstride=16,
                 input_fdim=128, input_tdim=input_tdim, model_size='base',
                 pretrain_stage=True)
    # # alternatively, create a frame based AST model
    # ast_mdl = ASTModel(
    #              fshape=128, tshape=2, fstride=128, tstride=2,
    #              input_fdim=128, input_tdim=input_tdim, model_size='base',
    #              pretrain=True)

    # do pretraining, see src/traintest_mask.py for our full pretraining code
    # input in shape [batch_size, input_tdim, input_fdim]
    test_input = torch.zeros([10, input_tdim, 128])
    # mask 100 patches for both discriminative and generative loss
    acc, nce_loss = ast_mdl(test_input, task='pretrain_mpc', mask_patch=100)
    mse_loss = ast_mdl(test_input, task='pretrain_mpg', mask_patch=100)
    loss = nce_loss + 10 * mse_loss
    # do back propagate and update the model, etc

    # after pretraining, save the pretrained model.
    # the code is designed for Dataparallel model
    ast_mdl = torch.nn.DataParallel(ast_mdl)
    torch.save(ast_mdl.state_dict(), './test_mdl.pth')

    # fine-tuning stage
    # now you have a labeled dataset you want to finetune AST on
    # suppose the avg length is 100 frames (1s) and there are 35 classes
    # the fshape and tshape must be same in pretraining and finetuning
    # but fstride and tstride can be different in pretraining and finetuning
    # using smaller strides improves the performance but also increase the computational overhead
    # set pretrain_stage as False since now is in the finetuning stage
    # provide the path of the pretrained model you want to load
    input_tdim = 100  # fine-tuning data length can be different with pretraining data length
    ast_mdl = ASTModel(label_dim=35,
                 fshape=16, tshape=16, fstride=10, tstride=10,
                 input_fdim=128, input_tdim=input_tdim, model_size='base',
                 pretrain_stage=False, load_pretrained_mdl_path='./test_mdl.pth')
    # # alternatively, use a frame based AST model
    # ast_mdl = ASTModel(label_dim=35,
    #              fshape=128, tshape=2, fstride=128, tstride=1,
    #              input_fdim=128, input_tdim=input_tdim, model_size='base',
    #              pretrain_stage=False, load_pretrained_mdl_path='./test_mdl.pth')

    # do finetuning, see src/traintest.py for our finetuning code
    test_input = torch.zeros([10, input_tdim, 128])
    prediction = ast_mdl(test_input, task='ft_avgtok')
    # output should in shape [batch_size, label_dim]
    print(prediction.shape)
    # calculate the loss, do back propagate, etc

    # # (optional) do some probe test
    # test_input = torch.zeros([1, input_tdim, 128]).to(device)
    # acc, nce = ast_mdl(test_input, task='pretrain_mpc', mask_patch=100)
    # # you can visualize the mask
    # pred, masked = ast_mdl(test_input, task='visualize_mask', mask_patch=100)
    # plt.imshow(masked[0,0])
    # plt.show()
    