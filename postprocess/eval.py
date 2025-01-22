###############################
#########  UNA Demo  ##########
###############################

from collections import defaultdict
import os, sys, warnings, shutil, time, datetime
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import math
import numpy as np
import torch
import torch.nn as nn 
from pytorch_msssim import ssim 

from utils.misc import make_dir, viewVolume, MRIread
import utils.test_utils as utils 
 
from Generator.constants import n_pathology, pathology_paths, pathology_prob_paths, \
    n_neutral_labels, label_list_segmentation, augmentation_funcs, processing_funcs, \
    dataset_setups, synth_dataset_setups


device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
exclude_keys = ['segmentation']
task_ckp_path = None


dataset_setups.update(synth_dataset_setups)

###############################
###############################
###############################

# prepare_one_hot

n_labels = len(label_list_segmentation)
label_list_segmentation = label_list_segmentation

lut = torch.zeros(10000, dtype=torch.long, device=device)
for l in range(n_labels):
    lut[label_list_segmentation[l]] = l
onehotmatrix = torch.eye(n_labels, dtype=torch.float, device=device) 
 

def get_l1(output, target, nonzero_only=False, *kwargs):
    if nonzero_only: # compute only within face_aware_region #
        nonzero_mask = target!=0
        l1 = (abs(target - output) * nonzero_mask).sum(dim=0) / nonzero_mask.sum(dim=0) 
    else:
        l1 = abs(target - output).mean() 
    return l1.cpu().numpy()
    
def get_psnr(output, target, *kwargs): 
    mse = (((output - target)**2).mean()).cpu().numpy()
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * math.log10(np.max(target.cpu().numpy()) / math.sqrt(mse))
    return psnr

def get_ssim(output, target, *kwargs):
    '''
    Ref: https://github.com/jorge-pessoa/pytorch-msssim
    ''' 
    ss = ssim(output, target, data_range = 1.0, size_average = False, win_sigma = 5.)
    return ss.mean().cpu().numpy()
 
def get_dice(output, target, *kwargs):
    """
    Dice of segmentation
    """
    output_onehot = onehotmatrix[lut[output.long()]][:, 0].permute([0, 4, 1, 2, 3]) # (1, 1, s, r, c, n_labels) -> (1, s, r, c, n_labels) -> (1, n_labels, s, r, c)
    target_onehot = onehotmatrix[lut[target.long()]][:, 0].permute([0, 4, 1, 2, 3])
    dice = torch.mean((2.0 * ((output_onehot * target_onehot).sum(dim=[2, 3, 4]))
                        / torch.clamp((output_onehot + target_onehot).sum(dim=[2, 3, 4]), min=1e-5)))
    return dice.cpu().numpy()

def get_normalized_l2(output, target, *kwargs):
    w = torch.sum(output * target) / (torch.sum(output ** 2) + 1e-7)
    l2 = 0. + torch.sqrt( torch.sum( (w * output - target) ** 2 ) / (torch.sum(target ** 2) + 1e-7) )
    return l2.cpu().numpy()


###############################
###############################
###############################


#main_test_dir = '~/results/test'
main_test_dir = '~/results/test_synth'

eval_model_list = [
                 os.path.join(main_test_dir, 't1_1-1-1'),
                ]


eval_dir_list = []
for model_dir in eval_model_list:
    dataset_names = os.listdir(model_dir) 
    for dataset_name in dataset_names:
        if os.path.isdir(os.path.join(model_dir, dataset_name)):
            eval_dir_list.append(os.path.join(model_dir, dataset_name))


max_num_test = 10 # None


for main_dir in eval_dir_list:
    
    print('Evaluating:', main_dir)
    dataset_name = os.path.basename(main_dir)
    score_txt = open(os.path.join(main_dir, 'scores.txt'), 'w') 

    modalities = dataset_setups[dataset_name]['modalities']

    cases = []
    for case in os.listdir(main_dir):
        if max_num_test is not None and len(cases) >= max_num_test:
            break
        if 'scores' not in case and os.path.isdir(os.path.join(main_dir, case)):
            cases.append(case)
    cases.sort()
    print('Num of testing cases: %d\n' % len(cases))
    score_txt.write('Num of testing cases: %d\n\n' % len(cases))

    t1_scores = {'L1_synthtoT1': [], 'L1_T1toT1': [], 'L1_T2toT1': [], 'L1_CTtoT1': [], 'L1_FLAIRtoT1': [],
                    'PSNR_synthtoT1': [], 'PSNR_T1toT1': [], 'PSNR_T2toT1': [], 'PSNR_CTtoT1': [], 'PSNR_FLAIRtoT1': [],
                    'SSIM_synthtoT1': [], 'SSIM_T1toT1': [], 'SSIM_T2toT1': [], 'SSIM_CTtoT1': [], 'SSIM_FLAIRtoT1': [],} 
    flair_scores = {'L1_synthtoFLAIR': [], 'L1_T1toFLAIR': [], 'L1_T2toFLAIR': [], 'L1_CTtoFLAIR': [], 'L1_FLAIRtoFLAIR': [],
                    'PSNR_synthtoFLAIR': [], 'PSNR_T1toFLAIR': [], 'PSNR_T2toFLAIR': [], 'PSNR_CTtoFLAIR': [], 'PSNR_FLAIRtoFLAIR': [],
                    'SSIM_synthtoFLAIR': [], 'SSIM_T1toFLAIR': [], 'SSIM_T2toFLAIR': [], 'SSIM_CTtoFLAIR': [], 'SSIM_FLAIRtoFLAIR': []}
    

    start_time = time.time()
    for i, case in enumerate(cases):
        if os.path.isdir(os.path.join(main_dir, case)):
            print('Processing %s (%d/%d)' % (case, i+1, len(cases)))

            case_dir = os.path.join(main_dir, case)

            mask = MRIread(os.path.join(case_dir, case + '.brainmask.nii.gz'), im_only = True)
            mask = np.nan_to_num(mask) 
            mask = torch.tensor(np.squeeze(mask), dtype=torch.int, device=device)[None, None]

            if os.path.isfile(os.path.join(case_dir, case + '.pathol.nii.gz')):
                pathol = MRIread(os.path.join(case_dir, case + '.pathol.nii.gz'), im_only = True)
                pathol = np.nan_to_num(pathol) 
                pathol = torch.tensor(np.squeeze(pathol), dtype=torch.int, device=device)[None, None]
                pathol_mask = 1 - pathol
                viewVolume(pathol_mask, names = ['pathol_mask'], save_dir = case_dir)
            else:
                pathol_mask = None
            
            score_string = case + ':'

            for in_mod in modalities:
                test_dir = os.path.join(case_dir, 'input_' + in_mod) 
                if not os.path.isdir(test_dir):
                    continue

                print('Current input modality:', in_mod)
                    
                # Synth
                for out_mod in modalities:
                    if not (os.path.isfile(os.path.join(test_dir, 'out_' + out_mod + '.nii.gz')) and os.path.isfile(os.path.join(case_dir, case + '.' + out_mod + '.nii.gz'))):
                        continue

                    postfix = '_%sto%s' % (in_mod, out_mod)

                    gt, aff = MRIread(os.path.join(case_dir, case + '.' + out_mod + '.nii.gz'), im_only=False, dtype='float') 
                    gt = np.nan_to_num(gt)
                    gt = torch.tensor(np.squeeze(gt), dtype=torch.float32, device=device)  
                    gt -= torch.min(gt)
                    gt /= torch.max(gt)
                    gt = gt[None, None] * mask
    
                    pd, aff = MRIread(os.path.join(test_dir, 'out_' + out_mod + '.nii.gz'), im_only=False, dtype='float') 
                    pd = np.nan_to_num(pd)
                    pd = torch.tensor(np.squeeze(pd), dtype=torch.float32, device=device) 
                    pd = pd[None, None] * mask

                    if pathol_mask is not None:
                        gt = gt * pathol_mask
                        pd = pd * pathol_mask

                    l1_score = get_l1(pd, gt)
                    psnr_score = get_psnr(pd, gt)
                    ssim_score = get_ssim(pd, gt)
                    score_string += ' L1%s - %.5f;' % (postfix, l1_score) 
                    score_string += ' PSNR%s - %.5f;' % (postfix, psnr_score) 
                    score_string += ' SSIM%s - %.5f;' % (postfix, psnr_score)
                    
                    if 'T1' in out_mod: 
                        t1_scores['L1' + postfix].append(l1_score)
                        t1_scores['PSNR' + postfix].append(psnr_score) 
                        t1_scores['SSIM' + postfix].append(ssim_score) 
                    elif 'FLAIR' in out_mod: 
                        flair_scores['L1' + postfix].append(l1_score) 
                        flair_scores['PSNR' + postfix].append(psnr_score)  
                        flair_scores['SSIM' + postfix].append(ssim_score)  


            score_txt.write(score_string+'\n') 
            print('   ', score_string)


    # summarize
    score_txt.write('\nAverage scores:\n') 
    for scores in [t1_scores, flair_scores]:
        score_string = '' 
        for k, v in scores.items():
            v = np.array(v)
            score_string += '   %s - %.5f (%.5f);\n' % (k, np.nanmean(v), np.nanvar(v))
            print('Final %s: %.5f' % (k, np.nanmean(v))) 
        score_txt.write(score_string+'\n\n') 
    score_txt.close()


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time: {}'.format(total_time_str))