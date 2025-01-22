

"""
Submodule interface.
"""
import torch

from .backbone import build_backbone
from .criterion import *
from .evaluator import Evaluator
from .head import get_head
from .joiner import get_processors, get_joiner
import utils.misc as utils


#########################################

# some constants
label_list_segmentation = [0, 14, 15, 16, 24, 77, 85, 2, 3, 4, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 41,
                                    42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60]
n_neutral_labels = 7
n_labels = len(label_list_segmentation)
nlat = int((n_labels - n_neutral_labels) / 2.0)
vflip = np.concatenate([np.array(range(n_neutral_labels)),
                        np.array(range(n_neutral_labels + nlat, n_labels)),
                        np.array(range(n_neutral_labels, n_neutral_labels + nlat))]) 

                        
############################################
############# helper functions #############
############################################

def process_args(gen_args, train_args, task):
    """
    task options: feat-anat, feat-seg, feat-anat-seg, anat, seg, reg, sr, bf
    """
    gen_args.tasks = [key for (key, value) in vars(task).items() if value]

    gen_args.generator.size = gen_args.generator.size # update real sample size (if sample_size is downsampled)

    gen_args.label_list_segmentation_with_csf = gen_args.label_list_segmentation_with_csf
    gen_args.label_list_segmentation_without_csf = gen_args.label_list_segmentation_without_csf
    gen_args.n_labels_with_csf = len(gen_args.label_list_segmentation_with_csf)
    gen_args.n_labels_without_csf = len(gen_args.label_list_segmentation_without_csf)

    train_args.out_channels = {}
    train_args.output_names = []
    train_args.aux_output_names = []
    train_args.target_names = []
    if not 'contrastive' in gen_args.tasks: 
        if 'T1' in gen_args.tasks:  
            train_args.out_channels['T1'] = 2 if train_args.losses.uncertainty is not None else 1
            train_args.output_names += ['T1']
            train_args.target_names += ['T1']
            if train_args.losses.uncertainty is not None:
                train_args.aux_output_names += ['T1_sigma']
        if 'T2' in gen_args.tasks:  
            train_args.out_channels['T2'] = 2 if train_args.losses.uncertainty is not None else 1
            train_args.output_names += ['T2']
            train_args.target_names += ['T2']
            if train_args.losses.uncertainty is not None:
                train_args.aux_output_names += ['T2_sigma']
        if 'FLAIR' in gen_args.tasks:  
            train_args.out_channels['FLAIR'] = 2 if train_args.losses.uncertainty is not None else 1
            train_args.output_names += ['FLAIR']
            train_args.target_names += ['FLAIR']
            if train_args.losses.uncertainty is not None:
                train_args.aux_output_names += ['FLAIR_sigma']
        if 'CT' in gen_args.tasks:  
            train_args.out_channels['CT'] = 2 if train_args.losses.uncertainty is not None else 1
            train_args.output_names += ['CT']
            train_args.target_names += ['CT']
            if train_args.losses.uncertainty is not None: # TODO
                train_args.aux_output_names += ['CT_sigma']
        if 'bias_fields' in gen_args.tasks:  
            train_args.out_channels['bias_field_log'] = 2 if train_args.losses.uncertainty is not None else 1
            train_args.output_names += ['bias_field_log']
            train_args.target_names += ['bias_field_log']
        if 'segmentation' in gen_args.tasks:  
            train_args.out_channels['segmentation'] = gen_args.n_labels_with_csf 
            train_args.output_names += ['label']
            train_args.target_names += ['label']
        if 'distance' in gen_args.tasks:  
            train_args.out_channels['distance'] = 4
            train_args.output_names += ['distance']
            train_args.target_names += ['distance']
        if 'surface' in gen_args.tasks:  
            train_args.out_channels['surface'] = 8
            train_args.output_names += ['surface']
            train_args.target_names += ['surface']

        if 'pathology' in gen_args.tasks:
            train_args.out_channels['pathology'] = 1 
            train_args.output_names += ['pathology']
            train_args.target_names += ['pathology']

        if 'encode_anomaly' in gen_args.tasks and 'pathology' not in train_args.target_names:  
            train_args.target_names += ['pathology']

        if train_args.losses.implicit_pathol: # TODO
            train_args.output_names += ['implicit_pathol_orig']
            train_args.output_names += ['implicit_pathol_pred']
            
        assert len(train_args.output_names) > 0

    return gen_args, train_args

############################################
################ CRITERIONS ################
############################################

def get_evaluator(args, task, device):
    """
    task options: sr, seg, anat, reg
    """
    metric_names = []
    if 'T1' in task or 'T2' in task or 'FLAIR' in task or 'CT' in task:
        metric_names += ['feat_ssim', 'feat_ms_ssim', 'feat_l1']
    else:
        if 'T1' in task: # TODO
            metric_names += ['recon_l1', 'recon_psnr', 'recon_ssim', 'recon_ms_ssim']
        if 'super_resolution' in task:
            metric_names += ['sr_l1', 'sr_psnr', 'sr_ssim', 'sr_ms_ssim']
        if 'bias_fields' in task: 
            metric_names += ['bf_normalized_l2', 'bf_corrected_l1']
        if 'segmentation' in task:
            metric_names += ['seg_dice']
        if 'pathology' in task:
            metric_names += ['pathol_dice']
        
    assert len(metric_names) > 0

    evaluator = Evaluator(
        args = args,
        metric_names = metric_names, 
        device = device,
        )
        
    return evaluator



def get_criterion(gen_args, train_args, tasks, device, exclude_keys = []):
    """
    task options: sr, seg, anat, reg
    """
    loss_names = []
    weight_dict = {}

    if 'contrastive' in tasks: 
        loss_names += ['contrastive']
        weight_dict['loss_contrastive'] = train_args.weights.contrastive
        return SetCriterion(
            args = train_args,
            weight_dict = weight_dict,
            loss_names = loss_names, 
            device = device,
            )
    
    #print(' all  tasks:', tasks)

    for task in tasks:

        if 'T1' in task or 'T2' in task or 'FLAIR' in task or 'CT' in task: 
            name = task

            loss_names += [name]
            weight_dict.update({'loss_%s' % name: train_args.weights.image})
            if train_args.losses.structure_contrastive:
                loss_names += ['%s_contrastive' % name]
                weight_dict['loss_%s_contrastive' % name] = train_args.weights.structure_contrastive 

            if train_args.losses.image_grad:
                loss_names += ['%s_grad' % name]
                weight_dict['loss_%s_grad' % name] = train_args.weights.image_grad 

        if 'segmentation' in task:
            loss_names += ['seg_ce', 'seg_dice']
            weight_dict.update( {
                'loss_seg_ce': train_args.weights.seg_ce,
                'loss_seg_dice': train_args.weights.seg_dice,
            } )
        
        if 'bias_fields' in task:
            loss_names += ['bias_field_log']
            weight_dict.update( {
                'loss_bias_field_log': train_args.weights.bias_field_log, 
            } )

        if 'registration' in task:
            loss_names += ['reg', 'reg_grad']
            weight_dict['loss_reg'] = train_args.weights.reg
            weight_dict['loss_reg_grad'] = train_args.weights.reg_grad

        if 'surface' in task:
            loss_names += ['surface']
            weight_dict['loss_surface'] = train_args.weights.surface

        if 'distance' in task:
            loss_names += ['distance']
            weight_dict['loss_distance'] = train_args.weights.distance

        if 'pathology' in task:
            loss_names += ['pathol_ce', 'pathol_dice']
            weight_dict.update( {
                'loss_pathol_ce': train_args.weights.pathol_ce,
                'loss_pathol_dice': train_args.weights.pathol_dice,
            } )

    if train_args.losses.implicit_pathol: 
        loss_names += ['implicit_pathol_ce', 'implicit_pathol_dice']
        weight_dict.update( {
            'loss_implicit_pathol_ce': train_args.weights.implicit_pathol_ce,
            'loss_implicit_pathol_dice': train_args.weights.implicit_pathol_dice,
        } )
        
    assert len(loss_names) > 0


    #print(' losses:', loss_names)

    criterion = SetMultiCriterion(
        gen_args = gen_args,
        train_args = train_args,
        weight_dict = weight_dict,
        loss_names = loss_names, 
        device = device,
        )
        
    return criterion




def get_postprocessor(gen_args, train_args, outputs, target, feats, tasks):
    """
    output: list of output dict 
    feat: list of output dict from pre-trained feat extractor
    """
    
    if 'segmentation' in tasks and target is not None:
        target['label'] = torch.tensor(gen_args.label_list_segmentation_with_csf, 
                                        device = target['segmentation'].device)[torch.argmax(target['segmentation'], 1, keepdim = True)] # (b, n_labels, s, r, c) -> (b, s, r, c) 

    for i, output in enumerate(outputs): 

        if feats is not None:
            output.update({'feat': feats[i]['feat']}) 

        if 'distance' in tasks:
            output['distance'] = torch.clamp(output['distance'], min = - gen_args.max_surf_distance, max = gen_args.max_surf_distance)
        
        if 'segmentation' in tasks:
            #output['label'] = torch.tensor(args.base_generator.label_list_segmentation_with_csf, 
            #                             device = output['segmentation'].device)[torch.argmax(output['segmentation'][:, vflip], 1, keepdim = True)] # (b, n_labels, s, r, c) -> (b, s, r, c) 
            output['label'] = torch.tensor(gen_args.label_list_segmentation_with_csf, 
                                         device = output['segmentation'].device)[torch.argmax(output['segmentation'], 1, keepdim = True)] # (b, n_labels, s, r, c) -> (b, s, r, c) 
        
    return outputs, target


#############################################
################ OPTIMIZERS #################
#############################################


def build_optimizer(train_args, params_groups):
    if train_args.optimizer == "adam":
        return torch.optim.Adam(params_groups)  
    elif train_args.optimizer == "adamw":
        return torch.optim.AdamW(params_groups)  # to use with ViTs
    elif train_args.optimizer == "sgd":
        return torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif train_args.optimizer == "lars":
        return utils.LARS(params_groups)  # to use with convnet and large batches
    else:
        ValueError('optim type {args.optimizer.type} supported!')


def build_schedulers(train_args, itr_per_epoch, lr, min_lr):
    if train_args.lr_scheduler == "cosine":
        lr_scheduler = utils.cosine_scheduler(
            lr, # * (args.batch_size * utils.get_world_size()) / 256.,  # linear scaling rule
            min_lr,
            train_args.n_epochs, itr_per_epoch,
            warmup_epochs=train_args.warmup_epochs
        )
    elif train_args.lr_scheduler == "multistep":
        lr_scheduler = utils.multistep_scheduler(
            lr, 
            train_args.lr_drops, 
            train_args.n_epochs, itr_per_epoch, 
            warmup_epochs=train_args.warmup_epochs, 
            gamma=train_args.lr_drop_multi
            )  
    wd_scheduler = utils.cosine_scheduler(
        train_args.weight_decay, # set as 0 to disable it
        train_args.weight_decay_end,
        train_args.n_epochs, itr_per_epoch
        )
    return lr_scheduler, wd_scheduler


############################################
################## MODELS ##################
############################################


def build_model(gen_args, train_args, device = 'cpu'):
    gen_args, train_args = process_args(gen_args, train_args, task = gen_args.task)

    backbone = build_backbone(train_args, train_args.backbone, 
                              num_cond = len(train_args.condition.split('+')) if train_args.condition is not None else 0)
    head = get_head(train_args, train_args.task_f_maps, train_args.out_channels, True, -1)
    model = get_joiner(gen_args.tasks, backbone, head, device) 

    processors = get_processors(train_args, gen_args.tasks, device)

    criterion = get_criterion(gen_args, train_args, gen_args.tasks, device)
    criterion.to(device)

    model.to(device)
    postprocessor = get_postprocessor

    return gen_args, train_args, model, processors, criterion, postprocessor 



def build_twostage_inpaint_model(gen_args, train_args, device = 'cpu'): # two-stage inpainting
    gen_args, train_args = process_args(gen_args, train_args, task = gen_args.task)

    # stage-0: pathology mask prediction
    pathol_backbone = build_backbone(train_args, train_args.backbone.split('+')[0], num_cond = 0)
    pathol_head = get_head(train_args, train_args.task_f_maps, train_args.out_channels, True, -1, stage = 0)
    pathol_model = get_joiner(gen_args.tasks, pathol_backbone, pathol_head, device, postfix = '_pathol')
    pathol_processors = get_processors(train_args, ['pathology'], device) 

    # stage-1: pathology-mask-conditioned task prediction (inpainting)  
    task_backbone = build_backbone(train_args, train_args.backbone.split('+')[1], num_cond = 1)
    task_head = get_head(train_args, train_args.task_f_maps, train_args.out_channels, True, -1, stage = 1)
    task_model = get_joiner(gen_args.tasks, task_backbone, task_head, device, postfix = '_task')
    task_processors = get_processors(train_args, gen_args.tasks, device, exclude_keys = ['pathology'])

    criterion = get_criterion(gen_args, train_args, gen_args.tasks, device)
    criterion.to(device)

    pathol_model.to(device)
    task_model.to(device)
    postprocessor = get_postprocessor

    return gen_args, train_args, pathol_model, task_model, pathol_processors, task_processors, criterion, postprocessor

 