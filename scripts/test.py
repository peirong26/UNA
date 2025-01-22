###############################
####  UNA Inference  #####
###############################

import os, sys, warnings, shutil, glob, time, datetime
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict 

import torch
import numpy as np

from utils.misc import make_dir, viewVolume, MRIread
import utils.test_utils as utils 
from Generator.utils import fast_3D_interp_torch 
from Generator.constants import dataset_setups, synth_dataset_setups

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

exclude_keys = ['segmentation']



dataset_setups.update(synth_dataset_setups)


model_cfg = '~/cfgs/trainer/train/joint_test.yaml'
gen_cfg = '~/cfgs/generator/train/inpaint_test.yaml'  



#win_size = [192, 192, 192] 
win_size = [160, 160, 160] 

zero_crop_first = False
mask_output = False 



dataset_names = ['ADHD', 'HCP', 'AIBL', 'OASIS', 'ADNI', 'ADNI3', 'ATLAS', 'ISLES']
#dataset_names = ['ADNI3', 'ATLAS', 'ISLES']
#dataset_names = ['ATLAS', 'ISLES']
dataset_names = []

synth_dataset_names = ['orig-T1', 'orig-T2', 'orig-FLAIR', 'orig-CT', 'orig-Synth']
#synth_dataset_names = ['orig-Synth']
synth_dataset_names = []


synth_dataset_pathol_names = ['orig-T1-synthpathol', 'orig-T2-synthpathol', 'orig-FLAIR-synthpathol', 'orig-CT-synthpathol', 'orig-Synth-synthpathol']
#synth_dataset_pathol_names = []


dataset_names = dataset_names + synth_dataset_names + synth_dataset_pathol_names




max_num_test_dataset = None #1, None
max_num_per_dataset  = None #5, None





#main_save_dir = make_dir('~/results/test/', reset = False)
main_save_dir = make_dir('~/results/test_synth/', reset = False)

models = [
    ('t1', '~/results/inpaint-all-flip/t1-wpde/1021-1247/ckp/checkpoint_latest.pth'),
    #('t1fl', '~/results/inpaint-all-flip/t1flair-wpde/1021-1855/ckp/checkpoint_latest.pth'),
]

#spacing = [1.5, 1.5, 5] # [1, 1, 1], [1.5, 1.5, 5], [3, 3, 3], None 
#add_bf = False
setups = [
    ([1, 1, 1], False), 
    #([1, 1, 1], True),
    #([1.5, 1.5, 5], False), 
    #([1.5, 1.5, 5], True), 
]





all_start_time = time.time()
for postfix, ckp_path in models:

    for spacing, add_bf in setups:
        curr_postfix = postfix + '_BF' if add_bf else postfix
        curr_postfix += '_%s-%s-%s' % (str(spacing[0]), str(spacing[1]), str(spacing[2])) if spacing is not None else '_1-1-1' 
        
        save_dir = make_dir(os.path.join(main_save_dir, curr_postfix), reset = False) # TODO
        print('\nSave at: %s\n' % save_dir)

        for i, dataset_name in enumerate(dataset_names):

            if max_num_test_dataset is not None and i >= max_num_test_dataset:
                break
                
            dataset_save_dir = make_dir(os.path.join(save_dir, dataset_name), reset = False)
            data_root = dataset_setups[dataset_name]['root']
            modalities = dataset_setups[dataset_name]['modalities']
            pathology_type = dataset_setups[dataset_name]['pathology_type']

            split_file = open(os.path.join(data_root, 'test.txt'), 'r')
            subj_names = []
            for subj in split_file.readlines():
                subj_names.append(subj.strip())
            subj_names.sort()
            print('Num of testing subjects in %s (%d/%d): %d' % (dataset_name, i+1, len(dataset_names), len(subj_names)))

            
            start_time = time.time()
            for j, subj_name in enumerate(subj_names):

                #if j+1 < 22:
                #    continue

                if max_num_per_dataset is not None and j >= max_num_per_dataset:
                    break

                subj_dir = make_dir(os.path.join(dataset_save_dir, subj_name.split('.nii')[0]))
                print('Now testing: %s (%d/%d)' % (subj_name, j+1, len(subj_names)))

                if dataset_setups[dataset_name]['paths']['flip2orig'] == 'synth_flip2orig':
                    flip2orig = os.path.join(data_root, dataset_setups[dataset_name]['paths']['flip2orig'])
                else:
                    flip2orig = np.loadtxt(os.path.join(data_root, dataset_setups[dataset_name]['paths']['flip2orig'], subj_name.split('.nii')[0] + '.txt'), delimiter=' ')
                    
                if pathology_type is not None:
                    pathol_path = os.path.join(data_root, dataset_setups[dataset_name]['paths']['pathology'], subj_name)
                    if os.path.isfile(pathol_path):
                        try:
                            pathol, _, _, aff, crop_start, orig_shp = utils.prepare_image(pathol_path, win_size = win_size, zero_crop_first = zero_crop_first, spacing = spacing, rescale = False, im_only = False, is_label = True, device = device, flip2orig = None)
                            viewVolume(pathol, names = [subj_name.split('.nii')[0] + '.pathol'], save_dir = subj_dir)
                        except:
                            pathol = None
                            print('Exception raised when reading pathology')
                    else:
                        pathol = None

                # save all GT and test
                avail_mod = []
                for mod in modalities:
                    mod_path = os.path.join(data_root, dataset_setups[dataset_name]['paths'][mod], subj_name)
                    if not os.path.isfile(mod_path):
                        continue
                    
                    try:
                        gt, im, im_flip2orig, aff, crop_start, orig_shp = utils.prepare_image(mod_path, win_size = win_size, zero_crop_first = zero_crop_first, spacing = spacing, add_bf = add_bf, is_CT = 'CT' in mod, im_only = False, device = device, flip2orig = flip2orig)
                        viewVolume(gt, names = [subj_name.split('.nii')[0] + '.' + mod], save_dir = subj_dir)
                        viewVolume(im, names = [subj_name.split('.nii')[0] + '.' + mod + '.input'], save_dir = subj_dir)
                        viewVolume(im_flip2orig, names = [subj_name.split('.nii')[0] + '.' + mod + '.input_flip2orig'], save_dir = subj_dir)
                        #viewVolume(high_res, names = [subj_name.split('.nii')[0] + '.high_res'], save_dir = subj_dir)
                        avail_mod.append(mod)
                        print('  Modality available:', mod)
                    except:
                        print('  Modality not available:', mod)
    
                for mod in avail_mod:
                    test_dir = make_dir(os.path.join(subj_dir, 'input_' + mod))
                    im = utils.read_image(os.path.join(subj_dir, subj_name.split('.nii')[0] + '.' + mod + '.input.nii.gz'), is_label = False, device = device)
                    im_flip2orig = utils.read_image(os.path.join(subj_dir, subj_name.split('.nii')[0] + '.' + mod + '.input_flip2orig.nii.gz'), is_label = False, device = device)
                    outs = utils.evaluate_image(im[None, None], im_flip2orig[None, None], ckp_path = ckp_path, feature_only = False, device = device, gen_cfg = gen_cfg, model_cfg = model_cfg)

                    if not os.path.isfile(os.path.join(subj_dir, subj_name.split('.nii')[0] + '.brainmask.nii.gz')):
                        mask = im.clone()
                        mask[im != 0.] = 1. 
                        viewVolume(mask, names = [subj_name.split('.nii')[0] + '.brainmask'], save_dir = subj_dir)

                    for k, v in outs.items(): 
                        if 'feat' not in k and k not in exclude_keys:
                            viewVolume(v * mask if mask_output else v, names = [ 'out_' + k], save_dir = test_dir)
            
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Testing time for {}: {}'.format(total_time_str, dataset_name))
            
        all_total_time = time.time() - all_start_time
        all_total_time_str = str(datetime.timedelta(seconds=int(all_total_time)))
print('Total testing time: {}'.format(total_time_str))
#'''