import tqdm
import math

import torch
import torch.nn as nn

from lib.utils.loss import (
    get_l2_loss, 
    get_distance_map_loss, 
    get_relative_orientation_loss, 
)
from lib.utils.proc_output import get_hand_obj_dist_map

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)

class Diffusion(nn.Module):
    def __init__(
            self, lhand_layer, rhand_layer, 
            beta_1=1e-4, beta_T=0.02, T=1000, 
            schedule_name="cosine", 
        ):
        super().__init__()

        self.lhand_layer = lhand_layer
        self.rhand_layer = rhand_layer
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        if schedule_name == "linear":
            betas = torch.linspace(start = beta_1, end=beta_T, steps=T)
        elif schedule_name == "cosine":
            betas = betas_for_alpha_bar(
                T,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )

        alphas = 1 - betas
        alpha_bars = torch.cumprod(
            alphas, 
            dim = 0
        )

        alpha_prev_bars = torch.cat([torch.Tensor([1]), alpha_bars[:-1]])
        sigmas = torch.sqrt((1 - alpha_prev_bars) / (1 - alpha_bars)) * torch.sqrt(1 - (alpha_bars / alpha_prev_bars))

        posterior_variance = (
            betas * (1.0 - alpha_prev_bars) / (1.0 - alpha_bars)
        )
        posterior_log_variance_clipped = torch.log(
            torch.hstack([posterior_variance[1], posterior_variance[1:]])
        )
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_prev_bars", alpha_prev_bars)
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)

    def forward(
        self, model, x_lhand, x_rhand, 
        x_obj, obj_feat, 
        timesteps=None, enc_text=None, 
        get_target=False, get_losses=False, 
        valid_mask_lhand=None, 
        valid_mask_rhand=None, 
        valid_mask_obj=None, 
        ldist_map=None, 
        rdist_map=None, 
        obj_verts_org=None, 
        loss_lambda_dict=None, 
        dataset_name=None, 
        obj_pc_top_idx=None
    ):
        assert enc_text is not None
        return_list = []

        if timesteps == None:
            timesteps = torch.randint(0, len(self.alpha_bars), (x_obj.size(0), )).to(x_obj.device)
            used_alpha_bars = self.alpha_bars[timesteps][:, None, None]
            epsilon_lhand = torch.randn_like(x_lhand)
            x_tilde_lhand = torch.sqrt(used_alpha_bars) * x_lhand + torch.sqrt(1 - used_alpha_bars) * epsilon_lhand
            epsilon_rhand = torch.randn_like(x_rhand)
            x_tilde_rhand = torch.sqrt(used_alpha_bars) * x_rhand + torch.sqrt(1 - used_alpha_bars) * epsilon_rhand
            epsilon_obj = torch.randn_like(x_obj)
            x_tilde_obj = torch.sqrt(used_alpha_bars) * x_obj + torch.sqrt(1 - used_alpha_bars) * epsilon_obj
            
        else:
            timesteps = torch.Tensor([timesteps for _ in range(x_obj.size(0))]).to(x_obj.device).long()
            x_tilde_lhand = x_lhand
            x_tilde_rhand = x_rhand
            x_tilde_obj = x_obj


        pred_X0_lhand, \
        pred_X0_rhand, \
        pred_X0_obj \
            = model(
                x_tilde_lhand, x_tilde_rhand, 
                x_tilde_obj, obj_feat, 
                timesteps, enc_text, 
                valid_mask_lhand,
                valid_mask_rhand, 
                valid_mask_obj
            )
        return_list.append(pred_X0_lhand)
        return_list.append(pred_X0_rhand)
        return_list.append(pred_X0_obj)
        if get_losses:
            target_lhand = x_lhand.clone().detach()
            target_rhand = x_rhand.clone().detach()
            target_obj = x_obj.clone().detach()
            total_loss = self.get_loss(
                timesteps, 
                pred_X0_lhand, pred_X0_rhand, pred_X0_obj, 
                target_lhand, target_rhand, target_obj, 
                valid_mask_lhand, valid_mask_rhand, valid_mask_obj, 
                ldist_map, rdist_map, obj_verts_org, 
                loss_lambda_dict, dataset_name, obj_pc_top_idx, 
            )
            return_list.append(total_loss)

        if get_target:
            return_list.append(epsilon_lhand)
            return_list.append(epsilon_rhand)
            return_list.append(epsilon_obj)
            return_list.append(used_alpha_bars)
        return return_list
    
    def get_loss(
            self, timesteps, 
            pred_X0_lhand, pred_X0_rhand, pred_X0_obj, 
            targ_lhand, targ_rhand, targ_obj, 
            valid_mask_lhand, valid_mask_rhand, valid_mask_obj, 
            ldist_map, rdist_map, obj_verts_org, 
            loss_lambda_dict, dataset_name, obj_pc_top_idx=None, 
        ):
        assert (
            valid_mask_lhand is not None and
            valid_mask_rhand is not None and
            valid_mask_obj is not None
        )

        # Loss weight
        # loss_weight = self.alpha_bars[timesteps]

        lambda_simple = loss_lambda_dict["lambda_simple"]
        lambda_dist = loss_lambda_dict["lambda_dist"]
        lambda_ro = loss_lambda_dict["lambda_ro"]
        
        # diffusion simple loss
        if lambda_simple > 0:
            simple_loss = get_l2_loss(
                pred_X0_lhand, pred_X0_rhand, pred_X0_obj, 
                targ_lhand, targ_rhand, targ_obj, 
                valid_mask_lhand, valid_mask_rhand, valid_mask_obj, 
            )
        else:
            simple_loss = torch.FloatTensor(1).fill_(0).cuda()

        # Distance map loss
        if lambda_dist > 0:
            pred_ldist, pred_rdist \
                = get_hand_obj_dist_map(
                    pred_X0_lhand, 
                    pred_X0_rhand, 
                    pred_X0_obj, 
                    obj_verts_org, 
                    self.lhand_layer, 
                    self.rhand_layer, 
                    dataset_name, 
                    obj_pc_top_idx, 
                )
            dist_map_loss = get_distance_map_loss(
                pred_ldist, pred_rdist,
                ldist_map, rdist_map, 
            )
        else:
            dist_map_loss = torch.FloatTensor(1).fill_(0).cuda()

        # Relative orientation loss
        if lambda_ro > 0:
            ro_loss = get_relative_orientation_loss(
                pred_X0_lhand, pred_X0_rhand, pred_X0_obj, 
                targ_lhand, targ_rhand, targ_obj, 
                valid_mask_lhand, valid_mask_rhand, 
            )
        else:
            ro_loss = torch.FloatTensor(1).fill_(0).cuda()
        
        total_loss = {
            "simple_loss": simple_loss,
            "dist_map_loss": dist_map_loss,
            "ro_loss": ro_loss,
        }
        return total_loss
    
    @torch.no_grad()
    def sampling(
        self, 
        model, obj_feat, 
        enc_text, enc_none_text, max_nframes, 
        hand_nfeats, obj_nfeats,
        valid_mask_lhand, 
        valid_mask_rhand, 
        valid_mask_obj, 
        device, 
        return_middle=False, 
        guidance_rate=7.5
    ):
        sampling_number = len(enc_text)
        sample_lhand = torch.randn([sampling_number, max_nframes, hand_nfeats]).to(device)
        sample_rhand = torch.randn([sampling_number, max_nframes, hand_nfeats]).to(device)
        sample_obj = torch.randn([sampling_number, max_nframes, obj_nfeats]).to(device)

        sample_lhand, sample_rhand, sample_obj = self.ddpm_loop(
            sample_lhand, sample_rhand, sample_obj, 
            model, obj_feat, enc_text, enc_none_text,
            valid_mask_lhand, valid_mask_rhand, valid_mask_obj, 
            return_middle, guidance_rate
        )
        return sample_lhand, sample_rhand, sample_obj
    
    def ddpm_loop(
            self, sample_lhand, sample_rhand, sample_obj, 
            model, obj_feat, enc_text, enc_none_text,
            valid_mask_lhand, valid_mask_rhand, valid_mask_obj, 
            return_middle, 
            guidance_rate
        ):
        for t_idx in tqdm.tqdm(
            reversed(range(len(self.alpha_bars))), 
            desc="sampling",
            total=len(self.alpha_bars)
        ):
            noise_lhand = torch.zeros_like(sample_lhand) if t_idx == 0 else torch.randn_like(sample_lhand)
            noise_rhand = torch.zeros_like(sample_rhand) if t_idx == 0 else torch.randn_like(sample_rhand)
            noise_obj = torch.zeros_like(sample_obj) if t_idx == 0 else torch.randn_like(sample_obj)

            pred_X0_lhand_cond, pred_X0_rhand_cond, pred_X0_obj_cond \
                = self.forward(
                    model, sample_lhand, sample_rhand, 
                    sample_obj, obj_feat, 
                    timesteps=t_idx, enc_text=enc_text, 
                    valid_mask_lhand=valid_mask_lhand, 
                    valid_mask_rhand=valid_mask_rhand, 
                    valid_mask_obj=valid_mask_obj
                )
            
            pred_X0_lhand_uncond, pred_X0_rhand_uncond, pred_X0_obj_uncond \
                = self.forward(
                    model, sample_lhand, sample_rhand, 
                    sample_obj, torch.zeros_like(obj_feat), 
                    timesteps=t_idx, enc_text=enc_none_text, 
                    valid_mask_lhand=valid_mask_lhand, 
                    valid_mask_rhand=valid_mask_rhand, 
                    valid_mask_obj=valid_mask_obj
                )
            pred_X0_lhand = pred_X0_lhand_uncond + guidance_rate * (pred_X0_lhand_cond - pred_X0_lhand_uncond)
            pred_X0_rhand = pred_X0_rhand_uncond + guidance_rate * (pred_X0_rhand_cond - pred_X0_rhand_uncond)
            pred_X0_obj = pred_X0_obj_uncond + guidance_rate * (pred_X0_obj_cond - pred_X0_obj_uncond)


            beta = self.betas[t_idx]
            alpha = self.alphas[t_idx]
            alpha_prev_bar = self.alpha_prev_bars[t_idx]
            alpha_bar = self.alpha_bars[t_idx]
            log_variance = self.posterior_log_variance_clipped[t_idx]

            coefficient_X0 = (beta*torch.sqrt(alpha_prev_bar)/(1-alpha_bar))
            coefficient_noise = ((1-alpha_prev_bar)*torch.sqrt(alpha)/(1-alpha_bar))

            mu_xt_lhand = pred_X0_lhand*coefficient_X0+sample_lhand*coefficient_noise
            mu_xt_rhand = pred_X0_rhand*coefficient_X0+sample_rhand*coefficient_noise
            mu_xt_obj = pred_X0_obj*coefficient_X0+sample_obj*coefficient_noise
            
            sample_lhand = mu_xt_lhand + torch.exp(0.5*log_variance) * noise_lhand
            sample_rhand = mu_xt_rhand + torch.exp(0.5*log_variance) * noise_rhand
            sample_obj = mu_xt_obj + torch.exp(0.5*log_variance) * noise_obj
            if return_middle and t_idx == 500:
                return sample_lhand, sample_rhand, sample_obj
        return sample_lhand, sample_rhand, sample_obj
    

    def sampling_loss_guidance(
        self, 
        model, obj_feat, 
        enc_none_text, max_nframes, 
        hand_nfeats, obj_nfeats,
        valid_mask_lhand, 
        valid_mask_rhand, 
        valid_mask_obj, 
        device, 
        return_middle=False, 
        ppx=None
    ):
        sampling_number = len(enc_none_text)
        sample_lhand = torch.randn([sampling_number, max_nframes, hand_nfeats]).to(device)
        sample_rhand = torch.randn([sampling_number, max_nframes, hand_nfeats]).to(device)
        sample_obj = torch.randn([sampling_number, max_nframes, obj_nfeats]).to(device)

        sample_lhand, sample_rhand, sample_obj = self.ddpm_loop_loss_guidance(
            sample_lhand, sample_rhand, sample_obj, 
            model, obj_feat, enc_none_text, 
            valid_mask_lhand, valid_mask_rhand, valid_mask_obj, 
            return_middle, ppx
        )
        return sample_lhand, sample_rhand, sample_obj
    
    def inpainting_loss(self, ppx, pred_X0_lhand_uncond, pred_X0_rhand_uncond):
        # loss = torch.norm(pred_X0_lhand_uncond[0,:] - ppx[0][0])
        # loss += torch.norm(pred_X0_rhand_uncond[0,:] - ppx[0][1])
        # loss += torch.norm(pred_X0_lhand_uncond[-1,:] - ppx[1][0])
        # loss += torch.norm(pred_X0_rhand_uncond[-1,:] - ppx[1][1])
        loss = (torch.norm(pred_X0_lhand_uncond[0,0,:] - ppx[0][0]) +
                torch.norm(pred_X0_rhand_uncond[0,0,:] - ppx[0][1]) +
                torch.norm(pred_X0_lhand_uncond[0,-1,:] - ppx[1][0])+
                torch.norm(pred_X0_rhand_uncond[0,-1,:] - ppx[1][1]))

        return loss
    
    def ddpm_loop_loss_guidance(
            self, sample_lhand, sample_rhand, sample_obj, 
            model, obj_feat, enc_none_text,
            valid_mask_lhand, valid_mask_rhand, valid_mask_obj, 
            return_middle, ppx
        ):

        pbar = tqdm.tqdm(
            reversed(range(len(self.alpha_bars))), 
            desc="sampling",
            total=len(self.alpha_bars)
        )

        for t_idx in pbar:
            noise_lhand = torch.zeros_like(sample_lhand) if t_idx == 0 else torch.randn_like(sample_lhand)
            noise_rhand = torch.zeros_like(sample_rhand) if t_idx == 0 else torch.randn_like(sample_rhand)
            noise_obj = torch.zeros_like(sample_obj) if t_idx == 0 else torch.randn_like(sample_obj)

            sample_lhand = sample_lhand.requires_grad_()
            sample_rhand = sample_rhand.requires_grad_()
            
            pred_X0_lhand_uncond, pred_X0_rhand_uncond, pred_X0_obj_uncond \
                = self.forward(
                    model, sample_lhand, sample_rhand, 
                    sample_obj, torch.zeros_like(obj_feat), 
                    timesteps=t_idx, enc_text=enc_none_text, 
                    valid_mask_lhand=valid_mask_lhand, 
                    valid_mask_rhand=valid_mask_rhand, 
                    valid_mask_obj=valid_mask_obj
                )
            
            pred_X0_lhand = pred_X0_lhand_uncond 
            pred_X0_rhand = pred_X0_rhand_uncond 
            pred_X0_obj = pred_X0_obj_uncond 

            loss = self.inpainting_loss(ppx, pred_X0_lhand, pred_X0_rhand)
            pbar.set_postfix({'loss': loss.item()}, refresh=False)

            grad_lhand, grad_rhand = torch.autograd.grad(
                outputs=loss,
                inputs=[sample_lhand, sample_rhand],
                retain_graph=False  # 可以设为False因为一次性计算完了
            )           

            # grad_lhand = torch.autograd.grad(outputs=loss, inputs=sample_lhand)[0]
            grad_lhand_norm = torch.norm(grad_lhand)
            # grad_rhand = torch.autograd.grad(outputs=loss, inputs=sample_rhand)[0]
            grad_rhand_norm = torch.norm(grad_rhand)


            beta = self.betas[t_idx]
            alpha = self.alphas[t_idx]
            alpha_prev_bar = self.alpha_prev_bars[t_idx]
            alpha_bar = self.alpha_bars[t_idx]
            log_variance = self.posterior_log_variance_clipped[t_idx]

            coefficient_X0 = (beta*torch.sqrt(alpha_prev_bar)/(1-alpha_bar))
            coefficient_noise = ((1-alpha_prev_bar)*torch.sqrt(alpha)/(1-alpha_bar))

            mu_xt_lhand = pred_X0_lhand*coefficient_X0+sample_lhand*coefficient_noise
            mu_xt_rhand = pred_X0_rhand*coefficient_X0+sample_rhand*coefficient_noise
            mu_xt_obj = pred_X0_obj*coefficient_X0+sample_obj*coefficient_noise

            sigma_t = torch.exp(0.5*log_variance)
            r = torch.sqrt(torch.tensor([150 * 99])).to(pred_X0_lhand.device) * sigma_t
            guidance_rate = 0.2
            eps = 1e-8
            # print(f'r: {r} sigma_t epsilon: {torch.norm(sigma_t * noise_lhand)}')

            d_star_lhand = -r * grad_lhand / (grad_lhand_norm + eps)
            d_sample_lhand = sigma_t * noise_lhand
            mix_direction_lhand = d_sample_lhand + guidance_rate * (d_star_lhand - d_sample_lhand)
            mix_direction_norm_lhand = torch.norm(mix_direction_lhand)
            mix_step_lhand = mix_direction_lhand / (mix_direction_norm_lhand + eps) * r

            d_star_rhand = -r * grad_rhand / (grad_rhand_norm + eps)
            d_sample_rhand = sigma_t * noise_rhand
            mix_direction_rhand = d_star_rhand + guidance_rate * (d_star_rhand - d_sample_rhand)
            mix_direction_norm_rhand = torch.norm(mix_direction_rhand)
            mix_step_rhand = mix_direction_rhand / (mix_direction_norm_rhand + eps) * r
            
            # sample_lhand = mu_xt_lhand + torch.exp(0.5*log_variance) * noise_lhand
            # sample_rhand = mu_xt_rhand + torch.exp(0.5*log_variance) * noise_rhand
            sample_lhand = mu_xt_lhand + mix_step_lhand
            sample_rhand = mu_xt_rhand + mix_step_rhand
            sample_obj = mu_xt_obj + torch.exp(0.5*log_variance) * noise_obj

            sample_lhand = sample_lhand.detach()
            sample_rhand = sample_rhand.detach()

            if return_middle and t_idx == 500:
                return sample_lhand, sample_rhand, sample_obj
        return sample_lhand, sample_rhand, sample_obj
    
    def sampling_projection(
        self, 
        model, obj_feat, 
        enc_none_text, max_nframes, 
        hand_nfeats, obj_nfeats,
        valid_mask_lhand, 
        valid_mask_rhand, 
        valid_mask_obj, 
        device, 
        return_middle=False, 
        ppx=None
    ):
        sampling_number = len(enc_none_text)
        sample_lhand = torch.randn([sampling_number, max_nframes, hand_nfeats]).to(device)
        sample_rhand = torch.randn([sampling_number, max_nframes, hand_nfeats]).to(device)
        sample_obj = torch.randn([sampling_number, max_nframes, obj_nfeats]).to(device)

        sample_lhand, sample_rhand, sample_obj = self.ddpm_loop_projection(
            sample_lhand, sample_rhand, sample_obj, 
            model, obj_feat, enc_none_text, 
            valid_mask_lhand, valid_mask_rhand, valid_mask_obj, 
            return_middle, ppx
        )
        return sample_lhand, sample_rhand, sample_obj

    def ddpm_loop_projection(
            self, sample_lhand, sample_rhand, sample_obj, 
            model, obj_feat, enc_none_text,
            valid_mask_lhand, valid_mask_rhand, valid_mask_obj, 
            return_middle, ppx
        ):

        pbar = tqdm.tqdm(
            reversed(range(len(self.alpha_bars))), 
            desc="sampling",
            total=len(self.alpha_bars)
        )

        for t_idx in pbar:
            noise_lhand = torch.zeros_like(sample_lhand) if t_idx == 0 else torch.randn_like(sample_lhand)
            noise_rhand = torch.zeros_like(sample_rhand) if t_idx == 0 else torch.randn_like(sample_rhand)
            noise_obj = torch.zeros_like(sample_obj) if t_idx == 0 else torch.randn_like(sample_obj)
            
            pred_X0_lhand, pred_X0_rhand, pred_X0_obj \
                = self.forward(
                    model, sample_lhand, sample_rhand, 
                    sample_obj, torch.zeros_like(obj_feat), 
                    timesteps=t_idx, enc_text=enc_none_text, 
                    valid_mask_lhand=valid_mask_lhand, 
                    valid_mask_rhand=valid_mask_rhand, 
                    valid_mask_obj=valid_mask_obj
                )
            
            beta = self.betas[t_idx]
            alpha = self.alphas[t_idx]
            alpha_prev_bar = self.alpha_prev_bars[t_idx]
            alpha_bar = self.alpha_bars[t_idx]
            log_variance = self.posterior_log_variance_clipped[t_idx]

            coefficient_X0 = (beta*torch.sqrt(alpha_prev_bar)/(1-alpha_bar))
            coefficient_noise = ((1-alpha_prev_bar)*torch.sqrt(alpha)/(1-alpha_bar))

            mu_xt_lhand = pred_X0_lhand*coefficient_X0+sample_lhand*coefficient_noise
            mu_xt_rhand = pred_X0_rhand*coefficient_X0+sample_rhand*coefficient_noise
            mu_xt_obj = pred_X0_obj*coefficient_X0+sample_obj*coefficient_noise

            sample_lhand = mu_xt_lhand + torch.exp(0.5*log_variance) * noise_lhand
            sample_rhand = mu_xt_rhand + torch.exp(0.5*log_variance) * noise_rhand
            sample_obj = mu_xt_obj + torch.exp(0.5*log_variance) * noise_obj

            def add_noise(x_0, t_idx):
                # timesteps = torch.tensor([t_idx for _ in range(x_0.shape[0])]).cuda()
                # used_alpha_bars = self.alpha_bars[timesteps][:, None, None]
                used_alpha_bars = self.alpha_bars[t_idx]
                epsilon = torch.randn_like(x_0)
                return torch.sqrt(used_alpha_bars) * x_0 + torch.sqrt(1 - used_alpha_bars) * epsilon

            print(ppx[0][0].shape, sample_lhand[0,0,:].shape, add_noise(ppx[0][0], t_idx).shape)
            sample_lhand[0,0,:] = add_noise(ppx[0][0], t_idx)
            sample_rhand[0,0,:] = add_noise(ppx[0][1], t_idx)
            sample_lhand[0,-1,:] = add_noise(ppx[1][0], t_idx)
            sample_rhand[0,-1,:] = add_noise(ppx[1][1], t_idx)

            if return_middle and t_idx == 500:
                return sample_lhand, sample_rhand, sample_obj
            
        return sample_lhand, sample_rhand, sample_obj