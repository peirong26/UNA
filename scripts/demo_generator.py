###############################
####  Synthetic Data Demo  ####
###############################


import datetime
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time 

import numpy as np

import torch

import utils.misc as utils 
 

from Generator import build_datasets 



# default & gpu cfg # 
default_gen_cfg_file = '~/cfgs/generator/default.yaml' 
demo_gen_cfg_file = '~/cfgs/generator/test/demo_generator.yaml'


def map_back_orig(img, idx, shp):
    if idx is None or shp is None:
        return img
    if len(img.shape) == 3:
        img = img[None, None]
    elif len(img.shape) == 4:
        img = img[None]
    return img[:, :, idx[0]:idx[0] + shp[0], idx[1]:idx[1] + shp[1], idx[2]:idx[2] + shp[2]]


def generate(args):

    _, gen_args, _ = args
 
    if gen_args.device_generator:
        device = gen_args.device_generator
    elif torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'  
    print('device: %s' % device) 
    print('out_dir:', gen_args.out_dir)
    print("Start generating")

    utils.make_dir(gen_args.out_dir, reset = True)
    start_time = time.time()


    tasks = [key for (key, value) in vars(gen_args.task).items() if value]


    # ============ preparing data ... ============ 
    dataset_dict = build_datasets(gen_args, device = gen_args.device_generator if gen_args.device_generator is not None else device) 
    

    cnt = 0
    while cnt < gen_args.test_itr_limit:
        
        dataset_idx = np.random.randint(len(gen_args.dataset_names))
        dataset_name = gen_args.dataset_names[dataset_idx]
        dataset = dataset_dict[dataset_name] 

        dataset.mild_samples = gen_args.mild_samples
        dataset.all_samples = gen_args.all_samples 


        subj_idx = np.random.randint(len(dataset.names)) 
        subj_name = os.path.basename(dataset.names[subj_idx]).split('.nii')[0]


        (input_mode, pathol_mode, subjects, samples) = dataset.__getitem__(subj_idx)

        if 'aff' in subjects:
            aff = subjects['aff']
            shp = subjects['shp']
            loc_idx = subjects['loc_idx']
        else:
            aff = torch.eye((4))
            shp = loc_idx = None 



        if isinstance(subjects['pathology'], float):
            print('pathology not available (%.5f), skipping...' % subjects['pathology'])
            continue
        elif subjects['pathology'].mean() < 0.005:
            print('pathology too small (%.5f), skipping...' % subjects['pathology'].mean())
            continue
    
        sample = samples[0] 
        if torch.isnan(sample['input'].mean()):
            print('Sample value is NaN, skipping...')
            continue

        save_dir = utils.make_dir(os.path.join(gen_args.out_dir, dataset_name + '.' + subj_name + '.' + input_mode))


        cnt += 1
        print('Generating image (%d/%d): %s' % (cnt, gen_args.test_itr_limit, dataset_name + '.' + subj_name + '.' + input_mode))
        print('input_mode:', input_mode)
        print('pathol_mode:', pathol_mode)
        

        utils.viewVolume(sample['input'], aff, names = [input_mode + '_diseased'], save_dir = save_dir)


        if 'T1' in subjects:
            utils.viewVolume(subjects['T1'], aff, names = ['T1'], save_dir = save_dir)
        if 'T2' in subjects:
            utils.viewVolume(subjects['T2'], aff, names = ['T2'], save_dir = save_dir)
        if 'FLAIR' in subjects:
            utils.viewVolume(subjects['FLAIR'], aff, names = ['FLAIR'], save_dir = save_dir)
        if 'CT' in subjects:
            utils.viewVolume(subjects['CT'], aff, names = ['CT'], save_dir = save_dir) 
        if 'pathology' in tasks:
            print(subjects['pathology'].mean(), subjects['pathology_orig'].mean(), subjects['pathology_progress_all'].mean())
            utils.viewVolume(subjects['pathology'], aff, names = ['pathology'], save_dir = save_dir)
            utils.viewVolume(subjects['pathology_prob'], aff, names = ['pathology_prob'], save_dir = save_dir)
            if 'pathology_orig' in subjects:
                utils.viewVolume(subjects['pathology_orig'], aff, names = ['pathology_orig'], save_dir = save_dir)
                utils.viewVolume(subjects['pathology_prob_orig'], aff, names = ['pathology_prob_orig'], save_dir = save_dir)  
                for t in range(subjects['pathology_progress_all'].shape[0]):
                    utils.viewVolume(subjects['pathology_progress_all'][t], aff, names = ['pathology_progress_%d' % t], save_dir = save_dir)   

        if 'V_dict' in subjects:
            for k, v in subjects['V_dict'].items():
                utils.viewVolume(subjects['V_dict'][k], aff, names = ['pathology_augment_' + k], save_dir = save_dir)
            Vx, Vy, Vz = subjects['V_dict']['Vx'], subjects['V_dict']['Vy'], subjects['V_dict']['Vz']
            V_norm = (Vx ** 2 + Vy ** 2 + Vz * 2) ** 0.5
            #utils.viewVolume(V_norm, aff, names = ['pathology_augment_V_norm'], save_dir = save_dir)
            utils.nda2img(torch.stack([Vx, Vy, Vz]).permute([3, 2, 1, 0]), isVector = True, save_path = os.path.join(save_dir, 'pathology_augment_V.nii.gz'))


 
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Generation time {}'.format(total_time_str))


#####################################################################################


if __name__ == '__main__': 
    gen_args = utils.preprocess_cfg([default_gen_cfg_file, demo_gen_cfg_file])
    utils.launch_job(submit_cfg = None, gen_cfg = gen_args, train_cfg = None, func = generate)