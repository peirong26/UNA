"""
Criterion modules.
"""

import numpy as np
import torch
import torch.nn as nn

from Trainer.models.losses import GradientLoss, gaussian_loss, laplace_loss, l1_loss
from utils.misc import viewVolume

uncertainty_loss = {'gaussian': gaussian_loss, 'laplace': laplace_loss}


class SetCriterion(nn.Module):
    """ 
    This class computes the loss for UNA.
    """
    def __init__(self, gen_args, train_args, weight_dict, loss_names, device):
        """ Create the criterion.
        Parameters:
            args: general exp cfg
            weight_dict: dict containing as key the names of the losses and as values their
                         relative weight.
            loss_names: list of all the losses to be applied. See get_loss for list of
                    available loss_names.
        """
        super(SetCriterion, self).__init__()
        self.gen_args = gen_args
        self.train_args = train_args
        self.weight_dict = weight_dict
        self.loss_names = loss_names 
 
        self.mse = nn.MSELoss()

        self.loss_regression_type = train_args.losses.uncertainty if train_args.losses.uncertainty is not None else 'l1' 
        self.loss_regression = uncertainty_loss[train_args.losses.uncertainty] if train_args.losses.uncertainty is not None else l1_loss
        
        self.pathol_weights = train_args.weights.pathol

        self.grad = GradientLoss('l1')

        self.bflog_loss = nn.L1Loss() if train_args.losses.bias_field_log_type == 'l1' else self.mse

        self.structure_contrastive = train_args.losses.structure_contrastive

        self.temp_alpha = train_args.contrastive_temperatures.alpha
        self.temp_beta = train_args.contrastive_temperatures.beta
        self.temp_gamma = train_args.contrastive_temperatures.gamma
        
        # initialize weights

        weights_with_csf = torch.ones(gen_args.n_labels_with_csf).to(device)
        weights_with_csf[gen_args.label_list_segmentation_with_csf==77] = train_args.relative_weight_lesions # give (more) importance to lesions
        weights_with_csf = weights_with_csf / torch.sum(weights_with_csf)

        weights_without_csf = torch.ones(gen_args.n_labels_without_csf).to(device)
        weights_without_csf[gen_args.label_list_segmentation_without_csf==77] = train_args.relative_weight_lesions # give (more) importance to lesions
        weights_without_csf = weights_without_csf / torch.sum(weights_without_csf)

        self.weights_ce = weights_with_csf[None, :, None, None, None]
        self.weights_dice = weights_with_csf[None, :]
        self.weights_dice_sup = weights_without_csf[None, :] 

        self.csf_ind = torch.tensor(np.where(np.array(gen_args.label_list_segmentation_with_csf)==24)[0][0])
        self.csf_v = torch.tensor(np.concatenate([np.arange(0, self.csf_ind), np.arange(self.csf_ind+1, gen_args.n_labels_with_csf)]))  

        self.loss_map = {
            'seg_ce': self.loss_seg_ce,
            'seg_dice': self.loss_seg_dice,
            'pathol_ce': self.loss_pathol_ce,
            'pathol_dice': self.loss_pathol_dice,
            'implicit_pathol_ce': self.loss_implicit_pathol_ce,
            'implicit_pathol_dice': self.loss_implicit_pathol_dice,
            'implicit_aux_pathol_ce': self.loss_implicit_aux_pathol_ce,
            'implicit_aux_pathol_dice': self.loss_implicit_aux_pathol_dice, 

            'T1': self.loss_T1,
            'T1_grad': self.loss_T1_grad,
            'T1_contrastive': self.loss_T1_contrastive,
            'T2': self.loss_T2,
            'T2_grad': self.loss_T2_grad,
            'T2_contrastive': self.loss_T2_contrastive,
            'FLAIR': self.loss_FLAIR,
            'FLAIR_grad': self.loss_FLAIR_grad,
            'FLAIR_contrastive': self.loss_FLAIR_contrastive,
            'CT': self.loss_CT,
            'CT_grad': self.loss_CT_grad,  
            'CT_contrastive': self.loss_CT_contrastive,

            "distance": self.loss_distance,
            "surface": self.loss_surface, # TODO
            "bias_field_log": self.loss_bias_field_log,
            'supervised_seg': self.loss_supervised_seg, 
            'contrastive': self.loss_feat_contrastive, 
        }

    def loss_feat_contrastive(self, outputs, *kwargs):
        """
        outputs: [feat1, feat2]
        feat shape: (b, feat_dim, s, r, c)
        """
        feat1, feat2 = outputs[0]['feat'][-1], outputs[1]['feat'][-1]
        num = torch.sum(torch.exp(feat1 * feat2 / self.temp_alpha), dim = 1) 
        den = torch.zeros_like(feat1[:, 0]) 
        for i in range(feat1.shape[1]): 
            den1 = torch.exp(feat1[:, i] * feat2[:, i] / self.temp_beta)
            den2 = torch.exp((torch.sum(feat1[:, i][:, None] * feat1, dim = 1) - feat1[:, i] ** 2) / self.temp_gamma) 
            den += den1 + den2 
        loss_contrastive = torch.mean(- torch.log(num /den)) 
        return {'loss_contrastive': loss_contrastive}

    def loss_structure_contrastive(self, input, input_flip, output):
        """
        output: (:, 1, s, r, c)
        target_flip: (:, 1, s, r, c)
        """  
        num = torch.exp(output * input_flip / self.temp_alpha) 
        den1 = torch.exp(output * input_flip / self.temp_beta)
        den2 = torch.exp(input * output / self.temp_gamma) 
        loss_sc = torch.mean(- torch.log(num / (den1 + den2)))   
        return loss_sc
    
    def loss_seg_ce(self, outputs, targets, *kwargs):
        """
        Cross entropy of segmentation
        """
        loss_seg_ce = torch.mean(-torch.sum(torch.log(torch.clamp(outputs['segmentation'], min=1e-5)) * self.weights_ce * targets['segmentation'], dim=1)) 
        return {'loss_seg_ce': loss_seg_ce}

    def loss_seg_dice(self, outputs, targets, *kwargs):
        """
        Dice of segmentation
        """
        loss_seg_dice = torch.sum(self.weights_dice * (1.0 - 2.0 * ((outputs['segmentation'] * targets['segmentation']).sum(dim=[2, 3, 4])) 
                                                       / torch.clamp((outputs['segmentation'] + targets['segmentation']).sum(dim=[2, 3, 4]), min=1e-5)))
        return {'loss_seg_dice': loss_seg_dice}
    
    def loss_implicit_pathol_ce(self, outputs, targets, samples, *kwargs):
        """
        Cross entropy of pathology segmentation
        """
        if 'implicit_pathol_pred' in outputs:
            #loss_implicit_pathol_ce = torch.mean(-torch.sum(torch.log(torch.clamp(outputs['implicit_pathol_pred'], min=1e-5)) * self.weights_ce * samples['pathol'], dim=1)) 
            loss_implicit_pathol_ce = torch.mean(-torch.sum(torch.log(torch.clamp(outputs['implicit_pathol_pred'], min=1e-5)) * outputs['implicit_pathol_orig'], dim=1))
        else: # no GT image exists
            loss_implicit_pathol_ce = 0.
        return {'loss_implicit_pathol_ce': loss_implicit_pathol_ce}
    
    def loss_implicit_pathol_dice(self, outputs, targets, samples, *kwargs):
        """
        Dice of pathology segmentation
        """
        if 'implicit_pathol_pred' in outputs:
            #loss_implicit_pathol_dice = torch.sum(self.weights_dice * (1.0 - 2.0 * ((outputs['implicit_pathol_pred'] * samples['pathol']).sum(dim=[2, 3, 4])) 
            #                                               / torch.clamp((outputs['implicit_pathol_pred'] + samples['pathol']).sum(dim=[2, 3, 4]), min=1e-5)))
            loss_implicit_pathol_dice = torch.sum((1.0 - 2.0 * ((outputs['implicit_pathol_pred'] * outputs['implicit_pathol_orig']).sum(dim=[2, 3, 4])) 
                                                        / torch.clamp((outputs['implicit_pathol_pred'] + outputs['implicit_pathol_orig']).sum(dim=[2, 3, 4]), min=1e-5)))
        else:
            loss_implicit_pathol_dice = 0.
        return {'loss_implicit_pathol_dice': loss_implicit_pathol_dice}


    def loss_implicit_aux_pathol_ce(self, outputs, targets, samples, *kwargs):
        """
        Cross entropy of pathology segmentation
        """
        if 'implicit_aux_pathol_pred' in outputs:
            #loss_implicit_aux_pathol_ce = torch.mean(-torch.sum(torch.log(torch.clamp(outputs['implicit_aux_pathol_pred'], min=1e-5)) * self.weights_ce * samples['pathol'], dim=1))  
            loss_implicit_aux_pathol_ce = torch.mean(-torch.sum(torch.log(torch.clamp(outputs['implicit_aux_pathol_pred'], min=1e-5)) * self.weights_ce * outputs['implicit_aux_pathol_orig'], dim=1))  
        else:
            loss_implicit_aux_pathol_ce = 0.
        return {'loss_implicit_aux_pathol_ce': loss_implicit_aux_pathol_ce}
    
    def loss_implicit_aux_pathol_dice(self, outputs, targets, samples, *kwargs):
        """
        Dice of pathology segmentation
        """
        if 'implicit_aux_pathol_pred' in outputs:
            #loss_implicit_aux_pathol_dice = torch.sum(self.weights_dice * (1.0 - 2.0 * ((outputs['implicit_aux_pathol_pred'] * samples['pathol']).sum(dim=[2, 3, 4])) 
            #                                               / torch.clamp((outputs['implicit_aux_pathol_pred'] + samples['pathol']).sum(dim=[2, 3, 4]), min=1e-5))) 
            loss_implicit_aux_pathol_dice = torch.sum(self.weights_dice * (1.0 - 2.0 * ((outputs['implicit_aux_pathol_pred'] * outputs['implicit_aux_pathol_orig']).sum(dim=[2, 3, 4])) 
                                                        / torch.clamp((outputs['implicit_aux_pathol_pred'] + outputs['implicit_aux_pathol_orig']).sum(dim=[2, 3, 4]), min=1e-5)))  
        else:
            loss_implicit_aux_pathol_dice = 0.
        return {'loss_implicit_aux_pathol_dice': loss_implicit_aux_pathol_dice}

    def loss_surface(self, outputs, targets, *kwargs): 
        loss_surface = self.mse(outputs['surface'], targets['surface'])
        return {'loss_surface': loss_surface}
    
    def loss_distance(self, outputs, targets, *kwargs): 
        loss_distance = self.mse(torch.clamp(outputs['distance'], min = - self.gen_args.max_surf_distance, max = self.gen_args.max_surf_distance), targets['distance'])
        return {'loss_distance': loss_distance}
    

    def loss_pathol_ce(self, outputs, targets, samples, *kwargs):
        """
        Cross entropy of pathology segmentation
        """
        if 'pathology' in outputs and outputs['pathology'].shape == targets['pathology'].shape:
            loss_pathol_ce = torch.mean(-torch.sum(torch.log(torch.clamp(outputs['pathology'], min=1e-5)) * targets['pathology'], dim=1))
        else:
            loss_pathol_ce = 0.
        return {'loss_pathol_ce': loss_pathol_ce}
    
    def loss_pathol_dice(self, outputs, targets, samples, *kwargs):
        """
        Dice of pathology segmentation
        """
        if 'pathology' in outputs and outputs['pathology'].shape == targets['pathology'].shape:
            loss_pathol_dice = torch.sum((1.0 - 2.0 * ((outputs['pathology'] * targets['pathology']).sum(dim=[2, 3, 4])) 
                                                        / torch.clamp((outputs['pathology'] + targets['pathology']).sum(dim=[2, 3, 4]), min=1e-5)))
        else:
            loss_pathol_dice = 0.
        return {'loss_pathol_dice': loss_pathol_dice}
    

    def loss_T1(self, outputs, targets, samples, input_mode, pathol_mode, *kwargs):
        if pathol_mode == 'real' and 'pathology' in targets and targets['pathology'].shape == targets['T1'].shape:
            # mask out real diseased region when computing reconstruction loss
            weights = 1. - targets['pathology'] if targets['pathology'].shape == targets['T1'].shape else 1. 
        else:
            # extra weights for supervised inpainting
            weights = 1. + targets['pathology'] * self.pathol_weights if targets['pathology'].shape == targets['T1'].shape else 1.
        return {'loss_T1': self.loss_image(outputs['T1'], targets['T1'], outputs['T1_sigma'] if 'T1_sigma' in outputs else None, weights = weights)} 

    def loss_T1_contrastive(self, outputs, targets, samples, input_mode, pathol_mode, *kwargs):
        if 'pathology' in targets and isinstance(targets['pathology'], torch.Tensor) and targets['pathology'].shape == targets['T1'].shape: 
            # exclude abnomal in flip2orig 
            weights = 1. - targets['common_pathology'] if 'common_pathology' in targets else 1.
            return {'loss_T1_contrastive': self.loss_structure_contrastive(samples['input'] * targets['pathology'] * weights, 
                                                                        samples['input_flip'] * targets['pathology'] * weights, 
                                                                        outputs['T1'] * targets['pathology'] * weights)}
        return {'loss_T1_contrastive': 0.} 

    def loss_T1_grad(self, outputs, targets, samples, input_mode, pathol_mode, *kwargs):
        if pathol_mode == 'real' and 'pathology' in targets and targets['pathology'].shape == targets['T1'].shape:
            # mask out real diseased region when computing reconstruction loss
            weights = 1. - targets['pathology'] if targets['pathology'].shape == targets['T1'].shape else 1. 
        else:
            # extra weights for supervised inpainting
            weights = 1. + targets['pathology'] * self.pathol_weights if targets['pathology'].shape == targets['T1'].shape else 1.
        return {'loss_T1_grad': self.loss_image_grad(outputs['T1'], targets['T1'], weights)}
    
    def loss_T2(self, outputs, targets, samples, input_mode, pathol_mode, *kwargs):
        if pathol_mode == 'real' and 'pathology' in targets and targets['pathology'].shape == targets['T2'].shape:
            # mask out real diseased region when computing reconstruction loss
            weights = 1. - targets['pathology'] if targets['pathology'].shape == targets['T2'].shape else 1. 
        else:
            # extra weights for supervised inpainting
            weights = 1. + targets['pathology'] * self.pathol_weights if targets['pathology'].shape == targets['T2'].shape else 1.
        return {'loss_T2': self.loss_image(outputs['T2'], targets['T2'], outputs['T2_sigma'] if 'T2_sigma' in outputs else None, weights)}  

    def loss_T2_contrastive(self, outputs, targets, samples, input_mode, pathol_mode, *kwargs):
        if 'pathology' in targets and isinstance(targets['pathology'], torch.Tensor) and targets['pathology'].shape == targets['T2'].shape: 
            # exclude abnomal in flip2orig
            weights = 1. - targets['common_pathology'] if 'common_pathology' in targets else 1.
            return {'loss_T2_contrastive': self.loss_structure_contrastive(samples['input'] * targets['pathology'] * weights, 
                                                                        samples['input_flip'] * targets['pathology'] * weights, 
                                                                        outputs['T2'] * targets['pathology'] * weights)}
        return {'loss_T2_contrastive': 0.} 
    
    def loss_T2_grad(self, outputs, targets, samples, input_mode, pathol_mode, *kwargs):
        if pathol_mode == 'real' and 'pathology' in targets and targets['pathology'].shape == targets['T2'].shape:
            # mask out real diseased region when computing reconstruction loss
            weights = 1. - targets['pathology'] if targets['pathology'].shape == targets['T2'].shape else 1. 
        else: 
            # extra weights for supervised inpainting
            weights = 1. + targets['pathology'] * self.pathol_weights if targets['pathology'].shape == targets['T2'].shape else 1.
        return {'loss_T2_grad': self.loss_image_grad(outputs['T2'], targets['T2'], weights)}
    
    def loss_FLAIR(self, outputs, targets, samples, input_mode, pathol_mode, *kwargs):
        if pathol_mode == 'real' and 'pathology' in targets and targets['pathology'].shape == targets['FLAIR'].shape:
            # mask out real diseased region when computing reconstruction loss
            weights = 1. - targets['pathology'] if targets['pathology'].shape == targets['FLAIR'].shape else 1. 
        else:
            # extra weights for supervised inpainting
            weights = 1. + targets['pathology'] * self.pathol_weights if targets['pathology'].shape == targets['FLAIR'].shape else 1.
        return {'loss_FLAIR': self.loss_image(outputs['FLAIR'], targets['FLAIR'], outputs['FLAIR_sigma'] if 'FLAIR_sigma' in outputs else None, weights)}  
    
    def loss_FLAIR_contrastive(self, outputs, targets, samples, input_mode, pathol_mode, *kwargs):
        if 'pathology' in targets and isinstance(targets['pathology'], torch.Tensor) and targets['pathology'].shape == targets['FLAIR'].shape: 
            # exclude abnomal in flip2orig
            weights = 1. - targets['common_pathology'] if 'common_pathology' in targets else 1.
            return {'loss_FLAIR_contrastive': self.loss_structure_contrastive(samples['input'] * targets['pathology'] * weights, 
                                                                        samples['input_flip'] * targets['pathology'] * weights, 
                                                                        outputs['FLAIR'] * targets['pathology'] * weights)}
        return {'loss_FLAIR_contrastive': 0.} 
    
    def loss_FLAIR_grad(self, outputs, targets, samples, input_mode, pathol_mode, *kwargs):
        if pathol_mode == 'real' and 'pathology' in targets and targets['pathology'].shape == targets['FLAIR'].shape:
            # mask out diseased region
            weights = 1. - targets['pathology'] if targets['pathology'].shape == targets['FLAIR'].shape else 1. 
        else:
            # extra weights for supervised inpainting
            weights = 1. + targets['pathology'] * self.pathol_weights if targets['pathology'].shape == targets['FLAIR'].shape else 1.
        return {'loss_FLAIR_grad': self.loss_image_grad(outputs['FLAIR'], targets['FLAIR'], weights)}
    
    def loss_CT(self, outputs, targets, samples, input_mode, pathol_mode, *kwargs):
        if pathol_mode == 'real' and 'pathology' in targets and targets['pathology'].shape == targets['CT'].shape:
            # mask out diseased region
            weights = 1. - targets['pathology'] if targets['pathology'].shape == targets['CT'].shape else 1. 
        else:
            # extra weights for supervised inpainting
            weights = 1. + targets['pathology'] * self.pathol_weights if targets['pathology'].shape == targets['CT'].shape else 1.
        return {'loss_CT': self.loss_image(outputs['CT'], targets['CT'], outputs['CT_sigma'] if 'CT_sigma' in outputs else None, weights)} 
    
    def loss_CT_contrastive(self, outputs, targets, samples, input_mode, pathol_mode, *kwargs):
        if 'pathology' in targets and isinstance(targets['pathology'], torch.Tensor) and targets['pathology'].shape == targets['CT'].shape: 
            # exclude abnomal in flip2orig
            weights = 1. - targets['common_pathology'] if 'common_pathology' in targets else 1.
            return {'loss_CT_contrastive': self.loss_structure_contrastive(samples['input'] * targets['pathology'] * weights, 
                                                                        samples['input_flip'] * targets['pathology'] * weights, 
                                                                        outputs['CT'] * targets['pathology'] * weights)}
        return {'loss_CT_contrastive': 0.} 
    
    def loss_CT_grad(self, outputs, targets, samples, pathol_mode, *kwargs):
        if pathol_mode == 'real' and 'pathology' in targets and targets['pathology'].shape == targets['CT'].shape:
            # mask out diseased region
            weights = 1. - targets['pathology'] if targets['pathology'].shape == targets['CT'].shape else 1. 
        else:
            # extra weights for supervised inpainting
            weights = 1. + targets['pathology'] * self.pathol_weights if targets['pathology'].shape == targets['CT'].shape else 1.
        return {'loss_CT_grad': self.loss_image_grad(outputs['CT'], targets['CT'], weights)}
    

    def loss_image(self, output, target, output_sigma = None, weights = 1., *kwargs): 
        if output.shape == target.shape:
            if output_sigma:
                loss_image = self.loss_regression(output, output_sigma, target)
            else:
                loss_image = self.loss_regression(output, target, weights)
        else: 
            loss_image = 0.
        return loss_image
    
    def loss_image_grad(self, output, target, weights = 1., *kwargs):
        return self.grad(output, target, weights) if output.shape == target.shape else 0. 

    
    def loss_bias_field_log(self, outputs, targets, samples):
        bf_soft_mask = 1. - targets['segmentation'][:, 0]
        loss_bias_field_log = self.bflog_loss(outputs['bias_field_log'] * bf_soft_mask, samples['bias_field_log'] * bf_soft_mask)
        return {'loss_bias_field_log': loss_bias_field_log}
    
    def loss_supervised_seg(self, outputs, targets, *kwargs):
        """
        Supervised segmentation differences (for dataset_name == synth)
        """
        onehot_withoutcsf = targets['segmentation'].clone()
        onehot_withoutcsf = onehot_withoutcsf[:, self.csf_v, ...]
        onehot_withoutcsf[:, 0, :, :, :] = onehot_withoutcsf[:, 0, :, :, :] + targets['segmentation'][:, self.csf_ind, :, :, :]

        loss_supervised_seg = torch.sum(self.weights_dice_sup * (1.0 - 2.0 * ((outputs['supervised_seg'] * onehot_withoutcsf).sum(dim=[2, 3, 4])) 
                                                                 / torch.clamp((outputs['supervised_seg'] + onehot_withoutcsf).sum(dim=[2, 3, 4]), min=1e-5)))

        return {'loss_supervised_seg': loss_supervised_seg} 

    def get_loss(self, loss_name, outputs, targets, samples, input_mode, pathol_mode, *kwargs):
        assert loss_name in self.loss_map, f'do you really want to compute {loss_name} loss?'
        return self.loss_map[loss_name](outputs, targets, samples, input_mode, pathol_mode, *kwargs)

    def forward(self, outputs, targets, samples, input_mode, pathol_mode, *kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied,
                      see each loss' doc
        """
        # Compute all the requested losses
        losses = {}
        for loss_name in self.loss_names:
            losses.update(self.get_loss(loss_name, outputs, targets, samples, input_mode, pathol_mode, *kwargs))
        return losses
    


class SetMultiCriterion(SetCriterion):
    """ 
    This class computes the loss for UNA with a list of results as inputs.
    """
    def __init__(self, gen_args, train_args, weight_dict, loss_names, device):
        """ Create the criterion.
        Parameters:
            args: general exp cfg
            weight_dict: dict containing as key the names of the losses and as values their
                         relative weight.
            loss_names: list of all the losses to be applied. See get_loss for list of
                    available loss_names.
        """
        super(SetMultiCriterion, self).__init__(gen_args, train_args, weight_dict, loss_names, device)
        self.all_samples = gen_args.generator.all_samples

    def get_loss(self, loss_name, outputs_list, targets, samples_list, input_mode, pathol_mode):
        assert loss_name in self.loss_map, f'do you really want to compute {loss_name} loss?'
        total_loss = 0. 
        for i_sample, outputs in enumerate(outputs_list): 
            curr_loss = self.loss_map[loss_name](outputs, targets, samples_list[i_sample], input_mode, pathol_mode)
            total_loss += curr_loss['loss_' + loss_name]
        return {'loss_' + loss_name: total_loss / self.all_samples}
    
    def forward(self, outputs_list, targets, samples_list, input_mode, pathol_mode):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied,
                      see each loss' doc
        """
        # Compute all the requested losses
        losses = {} 
        for loss_name in self.loss_names:
            losses.update(self.get_loss(loss_name, outputs_list, targets, samples_list, input_mode, pathol_mode)) 
        return losses

