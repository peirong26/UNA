import os
import numpy as np
import torch

from Generator.utils import fast_3D_interp_torch
from Trainer.models import build_model
from utils.checkpoint import load_checkpoint 
import utils.misc as utils 


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# default & gpu cfg # 

submit_cfg_file = os.path.join(root_dir, 'cfgs/submit.yaml')
default_gen_cfg_file = os.path.join(root_dir, 'cfgs/generator/default.yaml')
default_train_cfg_file = os.path.join(root_dir, 'cfgs/trainer/default_train.yaml')
default_val_file = os.path.join(root_dir, 'cfgs/trainer/default_val.yaml')

gen_cfg_dir = os.path.join(root_dir, 'cfgs/generator/test')
train_cfg_dir = os.path.join(root_dir, 'cfgs/trainer/test')



def zero_crop(orig, tol = 0, crop_range_lst = None, return_loc = False):
    
    '''
    crop_range_lst: [[x0, y0, z0], [x1, y1, z1]]
    ''' 
    
    if crop_range_lst is None:
            
        # Mask of non-black pixels (assuming image has a single channel).
        mask = orig > tol

        # Coordinates of non-black pixels.
        coords = torch.argwhere(mask)

        # Bounding box of non-black pixels. 
        #print(coords.min(dim=0))
        x0, y0, z0 = coords.min(dim=0)[0]
        x1, y1, z1 = coords.max(dim=0)[0] + 1   # slices are exclusive at the top

        # Check the the bounding box #
        #print('    Cropping Slice  [%d, %d)' % (x0, x1))
        #print('    Cropping Row    [%d, %d)' % (y0, y1))
        #print('    Cropping Column [%d, %d)' % (z0, z1))
        
    else:
        
        [[x0, y0, z0], [x1, y1, z1]] = crop_range_lst

    cropped = orig[x0 : x1, y0 : y1, z0 : z1] 

    if return_loc:
        return cropped, [[x0, y0, z0], [x1, y1, z1]]
     
    return cropped #, [[x0, y0, z0], [x1, y1, z1]]



def center_crop(img, win_size = [220, 220, 220], zero_crop_first = False):
    # center crop
    if len(img.shape) == 4: 
        img = torch.permute(img, (3, 0, 1, 2)) # (move last dim to first)
        img = img[None]
        permuted = True
    else: 
        assert len(img.shape) == 3
        img = img[None, None]
        permuted = False

    orig_shp = img.shape[2:] # (1, d, s, r, c) 

    # first, crop all zeros -> get *actual* shape
    if zero_crop_first:
        print(' before zero croppping:', orig_shp)
        img = zero_crop(img[0, 0])[None, None]
        orig_shp = img.shape[2:]
        print(' after zero croppping:', orig_shp)


    if win_size is None:
        if permuted:
            return torch.permute(img, (0, 2, 3, 4, 1)), [0, 0, 0], orig_shp
        return img, [0, 0, 0], orig_shp
    
    elif orig_shp[0] > win_size[0] or orig_shp[1] > win_size[1] or orig_shp[2] > win_size[2]:
        crop_start = [ max((orig_shp[i] - win_size[i]), 0) // 2 for i in range(3) ]
        crop_img = img[ :, :, crop_start[0] : crop_start[0] + win_size[0], 
                   crop_start[1] : crop_start[1] + win_size[1], 
                   crop_start[2] : crop_start[2] + win_size[2]]
        #pad_img = torch.zeros((1, 1, win_size[0], win_size[1], win_size[2]), device = device)
        #pad_img[:, :, int((win_size[0] - crop_img.shape[2])/2) : int((win_size[0] - crop_img.shape[2])/2) + crop_img.shape[2], \
        #            int((win_size[1] - crop_img.shape[3])/2) : int((win_size[1] - crop_img.shape[3])/2) + crop_img.shape[3], \
        #            int((win_size[2] - crop_img.shape[4])/2) : int((win_size[2] - crop_img.shape[4])/2) + crop_img.shape[4] ] = crop_img 
        if permuted:
            return torch.permute(crop_img, (0, 2, 3, 4, 1)), [0, 0, 0], orig_shp
        return crop_img, crop_start, orig_shp
    
    else:
        #pad_img = torch.zeros((1, 1, win_size[0], win_size[1], win_size[2]), device = device)
        #pad_img[:, :, int((win_size[0] - img.shape[2])/2) : int((win_size[0] - img.shape[2])/2) + img.shape[2], \
        #            int((win_size[1] - img.shape[3])/2) : int((win_size[1] - img.shape[3])/2) + img.shape[3], \
        #            int((win_size[2] - img.shape[4])/2) : int((win_size[2] - img.shape[4])/2) + img.shape[4] ] = img 
        if permuted:
            return torch.permute(img, (0, 2, 3, 4, 1)), [0, 0, 0], orig_shp
        return img, [0, 0, 0], orig_shp




def center_crop_with_pad(img, win_size = [220, 220, 220], zero_crop_first = False):
    # center crop
    if len(img.shape) == 4: 
        img = torch.permute(img, (3, 0, 1, 2)) # (move last dim to first)
        img = img[None]
        permuted = True
    else: 
        assert len(img.shape) == 3
        img = img[None, None]
        permuted = False 

    orig_shp = img.shape[2:] # (1, d, s, r, c) 

    # first, crop all zeros -> get *actual* shape
    if zero_crop_first:
        print(' before zero croppping:', orig_shp)
        img = zero_crop(img[0, 0])[None, None]
        orig_shp = img.shape[2:]
        print(' after zero croppping:', orig_shp)


    if win_size is None:
        if permuted:
            return torch.permute(img, (0, 2, 3, 4, 1)), [0, 0, 0], orig_shp
        return img, [0, 0, 0], orig_shp
    
    elif orig_shp[0] > win_size[0] or orig_shp[1] > win_size[1] or orig_shp[2] > win_size[2]:
        crop_start = [ max((orig_shp[i] - win_size[i]), 0) // 2 for i in range(3) ]
        crop_img = img[ :, :, crop_start[0] : crop_start[0] + win_size[0], 
                   crop_start[1] : crop_start[1] + win_size[1], 
                   crop_start[2] : crop_start[2] + win_size[2]]
        
        pad_img = torch.zeros((1, 1, win_size[0], win_size[1], win_size[2]), device = img.device)
        pad_img[:, :, int((win_size[0] - crop_img.shape[2])/2) : int((win_size[0] - crop_img.shape[2])/2) + crop_img.shape[2], \
                    int((win_size[1] - crop_img.shape[3])/2) : int((win_size[1] - crop_img.shape[3])/2) + crop_img.shape[3], \
                    int((win_size[2] - crop_img.shape[4])/2) : int((win_size[2] - crop_img.shape[4])/2) + crop_img.shape[4] ] = crop_img 
        if permuted:
            return torch.permute(crop_img, (0, 2, 3, 4, 1)), [0, 0, 0], orig_shp
        return pad_img, crop_start, orig_shp
    
    else:
        pad_img = torch.zeros((1, 1, win_size[0], win_size[1], win_size[2]), device = img.device)
        pad_img[:, :, int((win_size[0] - img.shape[2])/2) : int((win_size[0] - img.shape[2])/2) + img.shape[2], \
                    int((win_size[1] - img.shape[3])/2) : int((win_size[1] - img.shape[3])/2) + img.shape[3], \
                    int((win_size[2] - img.shape[4])/2) : int((win_size[2] - img.shape[4])/2) + img.shape[4] ] = img 
        if permuted:
            return torch.permute(pad_img, (0, 2, 3, 4, 1)), [0, 0, 0], orig_shp
        return pad_img, [0, 0, 0], orig_shp
    

def deform_flip2orig(I_orig, aff_orig, flip2orig):

    if flip2orig is None:
        return I_orig

    I_flip = torch.flip(I_orig, [0])
    size = I_flip.shape

    xx, yy, zz = np.meshgrid(range(size[0]), range(size[1]), range(size[2]), sparse=False, indexing='ij')
    xx = torch.tensor(xx, dtype=torch.float, device=I_orig.device)
    yy = torch.tensor(yy, dtype=torch.float, device=I_orig.device)
    zz = torch.tensor(zz, dtype=torch.float, device=I_orig.device)


    # deform flip to non-flip space
    M = torch.linalg.inv(aff_orig) @ flip2orig @ aff_orig
    xx2 = M[0,0] * xx + M[0,1] * yy + M[0,2] * zz + M[0,3]
    yy2 = M[1,0] * xx + M[1,1] * yy + M[1,2] * zz + M[1,3]
    zz2 = M[2,0] * xx + M[2,1] * yy + M[2,2] * zz + M[2,3] 
    I_flip2orig = fast_3D_interp_torch(I_flip, xx2, yy2, zz2, 'linear')
    
    return I_flip2orig


def read_image(img_path, is_label = False, device = 'cpu'):
    im, aff = utils.MRIread(img_path, im_only=False, dtype='int' if is_label else 'float')
    im = torch.tensor(np.squeeze(im), dtype=torch.int if is_label else torch.float32, device=device)
    im = torch.nan_to_num(im)
    return im


def prepare_image(img_path, win_size = None, zero_crop_first = False, spacing = None, 
                    add_bf = False, is_CT = False, is_label = False, rescale = True, 
                    hemis_mask = None, im_only = False, device = 'cpu', flip2orig = None):

    im, aff = utils.MRIread(img_path, im_only=False, dtype='int' if is_label else 'float')
    im = torch.tensor(np.squeeze(im), dtype=torch.int if is_label else torch.float32, device=device)
    im = torch.nan_to_num(im)

    if len(im.shape) > 3:
        #print('shape', im.shape) 
        im = im.mean(dim = -1) # averaging the RGB

    if is_CT and rescale: # for CT as input
        im = torch.clamp(im, min = 0., max = 80.)

    im_flip2orig = None
    if flip2orig is not None:
        if os.path.isdir(flip2orig):
            im_flip2orig_path = os.path.join(flip2orig, os.path.basename(img_path))
            im_flip2orig, _ = utils.MRIread(im_flip2orig_path, im_only=False, dtype='int' if is_label else 'float')
            im_flip2orig = torch.tensor(np.squeeze(im_flip2orig), dtype=torch.int if is_label else torch.float32, device=device)
            im_flip2orig = torch.nan_to_num(im_flip2orig)
        else:
            flip2orig = torch.from_numpy(flip2orig).float().to(device) # 4x4
            im_flip2orig = deform_flip2orig(im, torch.tensor(aff, dtype=torch.float, device=im.device), flip2orig)

    if not is_label and rescale:
        im -= torch.min(im)
        im /= torch.max(im)
        if flip2orig is not None:
            im_flip2orig -= torch.min(im_flip2orig)
            im_flip2orig /= torch.max(im_flip2orig)

    im, aff = utils.torch_resize(im, aff, 1.)
    if flip2orig is not None:
        im_flip2orig, aff = utils.torch_resize(im_flip2orig, aff, 1.)

    gt = im
    gt, aff = utils.align_volume_to_ref(gt, aff, aff_ref=np.eye(4), return_aff=True, n_dims=3)
    gt, crop_start, orig_shp = center_crop(gt, win_size, zero_crop_first = zero_crop_first)

    if spacing is not None:
        im = resample(im, new_res = spacing)
        if flip2orig is not None:
            im_flip2orig = resample(im_flip2orig, new_res = spacing)

    im, aff = utils.align_volume_to_ref(im, aff, aff_ref=np.eye(4), return_aff=True, n_dims=3)
    im, crop_start, orig_shp = center_crop(im, win_size, zero_crop_first = zero_crop_first)
    if flip2orig is not None:
        im_flip2orig, aff = utils.align_volume_to_ref(im_flip2orig, aff, aff_ref=np.eye(4), return_aff=True, n_dims=3)
        im_flip2orig, crop_start, orig_shp = center_crop(im_flip2orig, win_size, zero_crop_first = zero_crop_first)

    if im_only:
        return gt, im, im_flip2orig, aff
    return gt, im, im_flip2orig, aff, crop_start, orig_shp


def resample(I, orig_res = [1., 1., 1.], new_res = [1., 1., 1.]):
    if not isinstance(orig_res, list):
        orig_res = [orig_res, orig_res, orig_res]
    if not isinstance(new_res, list):
        new_res = [new_res, new_res, new_res]
    #print('pre resample', I.shape)
    resolution = np.array(new_res)
    new_size = (np.array(I.shape) * orig_res / resolution).astype(int)

    factors = np.array(new_size) / np.array(I.shape)
    delta = (1.0 - factors) / (2.0 * factors)
    vx = np.arange(delta[0], delta[0] + new_size[0] / factors[0], 1 / factors[0])[:new_size[0]]
    vy = np.arange(delta[1], delta[1] + new_size[1] / factors[1], 1 / factors[1])[:new_size[1]]
    vz = np.arange(delta[2], delta[2] + new_size[2] / factors[2], 1 / factors[2])[:new_size[2]]
    II, JJ, KK = np.meshgrid(vx, vy, vz, sparse=False, indexing='ij')
    II = torch.tensor(II, dtype=torch.float, device=I.device)
    JJ = torch.tensor(JJ, dtype=torch.float, device=I.device)
    KK = torch.tensor(KK, dtype=torch.float, device=I.device)

    I_resize = fast_3D_interp_torch(I, II, JJ, KK, 'linear') 
    I_new = utils.myzoom_torch(I_resize, 1 / factors) 

    #print('post resample', I_new.shape)
    return I_new



@torch.no_grad()
def evaluate_image(inputs, inputs_flip, ckp_path, feature_only = False, device = 'cpu', gen_cfg = None, model_cfg = None):
    # inputs: Torch.Tensor -- (batch_size, 1, s, r, c)

    # ============ prepare ... ============
    gen_args = utils.preprocess_cfg([default_gen_cfg_file, gen_cfg], cfg_dir = gen_cfg_dir) 
    train_args = utils.preprocess_cfg([default_train_cfg_file, default_val_file, model_cfg], cfg_dir = train_cfg_dir)

    samples = [ { 'input': inputs } ]

    if inputs_flip is not None:
        cond = [inputs_flip]
    else:
        cond = []

    # ============ testing ... ============ 
    gen_args, train_args, feat_model, processors, criterion, postprocessor = build_model(gen_args, train_args, device) 
    load_checkpoint(ckp_path, [feat_model], model_keys = ['model'], to_print = False)

    outputs, _ = feat_model(samples, cond = cond) # dict with features 

    for processor in processors:
        outputs = processor(outputs, samples)
    if postprocessor is not None:
        outputs, _ = postprocessor(gen_args, train_args, outputs, target = None, feats = None, tasks = gen_args.tasks)

    if feature_only:
        return outputs[0]['feat'][-1] # (batch_size, 64, s, r, c)
    else:
        return outputs[0]
    

@torch.no_grad()
def evaluate_image_twostage(inputs, pathol_ckp_path, task_ckp_path, feature_only = True, device = 'cpu', gen_cfg = None, model_cfg = None):
    # inputs: Torch.Tensor -- (batch_size, 1, s, r, c)

    # ============ prepare ... ============
    gen_args = utils.preprocess_cfg([default_gen_cfg_file, gen_cfg], cfg_dir = gen_cfg_dir) 
    train_args = utils.preprocess_cfg([default_train_cfg_file, default_val_file, model_cfg], cfg_dir = train_cfg_dir)

    samples = [ { 'input': inputs } ]

    # ============ testing ... ============ 
    gen_args, train_args, pathol_model, task_model, pathol_processors, task_processors, criterion, postprocessor = build_inpaint_model(gen_args, train_args, device) 
    load_checkpoint(pathol_ckp_path, [pathol_model], model_keys = ['model'], to_print = False)
    load_checkpoint(task_ckp_path, [task_model], model_keys = ['model'], to_print = False)

    # stage-0: pathology segmentation prediction
    outputs_pathol, _ = pathol_model(samples)
    for processor in pathol_processors:
        outputs_pathol = processor(outputs_pathol, samples)

    # stage-1: pathology-mask-conditioned inpainting tasks prediction
    for i in range(len(samples)): # mask using predicted anomaly
        samples[i]['input_masked'] = samples[i]['input'] * (1 - outputs_pathol[i]['pathology'])
    outputs_task, _ = task_model(samples, input_name = 'input_masked', cond = [o['pathology'] for o in outputs_pathol])
    for processor in task_processors:
        outputs_task = processor(outputs_task, samples)

    outputs = utils.merge_list_of_dict(outputs_task, outputs_pathol) 

    if postprocessor is not None:
        outputs = postprocessor(gen_args, train_args, outputs, feats = None, tasks = gen_args.tasks)

    if feature_only:
        return outputs[0]['feat_pathol'][-1], outputs[0]['feat_task'][-1] # (batch_size, 64, s, r, c)
    else:
        return outputs[0]



@torch.no_grad()
def evaluate_path(input_paths, save_dir, ckp_path, win_size = [220, 220, 220], 
                  save_input = False, aux_paths = {}, save_aux = False, exclude_keys = [], 
                  mask_output = False, ext = '.nii.gz', device = 'cpu', 
                  gen_cfg = None, model_cfg = None):
     
    gen_args = utils.preprocess_cfg([default_gen_cfg_file, gen_cfg], cfg_dir = gen_cfg_dir) 
    train_args = utils.preprocess_cfg([default_train_cfg_file, default_val_file, model_cfg], cfg_dir = train_cfg_dir)
    

    # ============ loading ... ============
    gen_args, train_args, feat_model, processors, criterion, postprocessor = build_model(gen_args, train_args, device) 
    load_checkpoint(ckp_path, [feat_model], model_keys = ['model'], to_print = False) 

    for i, input_path in enumerate(input_paths):
        print('Now testing: %s (%d/%d)' % (input_path, i+1, len(input_paths)))
        print('        ckp:', ckp_path)
        curr_save_dir = utils.make_dir(os.path.join(save_dir, os.path.basename(input_path).split('.nii')[0]))

        # ============ prepare ... ============
        mask = None
        im, aff, crop_start, orig_shp = prepare_image(input_path, win_size, device = device)
        if save_input:
            print('  Input: saved in - %s' % (os.path.join(curr_save_dir, 'input' + ext)))
            utils.viewVolume(im, aff, names = ['input'], ext = ext, save_dir = curr_save_dir)
        for k in aux_paths.keys():
            im_k, _, _, _ = prepare_image(aux_paths[k][i], win_size, is_label = 'label' in k, device = device)
            if save_aux:
                print('  Aux input: %s - saved in - %s' % (k, os.path.join(curr_save_dir, k + ext)))
                utils.viewVolume(im_k, aff, names = [k], ext = ext, save_dir = curr_save_dir)
            if mask_output and 'mask' in k:
                mask = im_k.clone()
                mask[im_k != 0.] = 1.
        samples = [ { 'input': im } ]
    
        # ============ testing ... ============
        outputs, _ = feat_model(samples) # dict with features 

        for processor in processors:
            outputs = processor(outputs, samples)
        if postprocessor is not None:
            outputs = postprocessor(gen_args, train_args, outputs, feats = None, tasks = gen_args.tasks)

        out = outputs[0]
        if mask_output and mask is None:
            mask = torch.zeros_like(im)
            mask[im != 0.] = 1.
        for key in out.keys():
            if key not in exclude_keys and isinstance(out[key], torch.Tensor):
                print('  Output: %s - saved in - %s' % (key, os.path.join(curr_save_dir, 'out_' + key + ext)))
                out[key][out[key] < 0.] = 0.
                utils.viewVolume(out[key] * mask if mask_output else out[key], aff, names = ['out_'+key], ext = ext, save_dir = curr_save_dir)
