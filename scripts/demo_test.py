###############################
#########  UNA Demo  ##########
###############################

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import utils.test_utils as utils 
from utils.misc import viewVolume, make_dir

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_cfg = os.path.join(root_dir, 'cfgs/trainer/test/test.yaml')
gen_cfg = os.path.join(root_dir, 'cfgs/generator/test/test.yaml') 
ckp_path = os.path.join(root_dir, 'assets/una.pth') 

win_size = [160, 160, 160]



def test(case_dir, ckp_path, save_dir, win_size = None):

    ### Read Paths ###
    img_path = os.path.join(case_dir, 'input.nii.gz')
    img_flip_reg2orig_path = os.path.join(case_dir, 'input_flip_reg2orig.nii.gz')

    ### Input Preparation ### 
    _, img, _, aff = utils.prepare_image(img_path, win_size = win_size, im_only = True, device = device) 
    _, img_flip_reg2orig, _, _ = utils.prepare_image(img_flip_reg2orig_path, win_size = win_size, spacing = None, im_only = True, device = device)

    ### Inference ###
    outs = utils.evaluate_image(img, img_flip_reg2orig, ckp_path = ckp_path, device = device, gen_cfg = gen_cfg, model_cfg = model_cfg)
    viewVolume(outs['T1'], aff, names = [ 'out_una' ], save_dir = save_dir)




############################################################

if __name__ == '__main__': 

    main_save_dir = make_dir(os.path.join(root_dir, 'assets/results/'), reset = False)
    main_test_dir = os.path.join(root_dir, 'assets/data')
    case_names = os.listdir(main_test_dir)

    for case_name in case_names:
        print(case_name)
        case_data_dir = os.path.join(main_test_dir, case_name)
        case_save_dir = make_dir(os.path.join(main_save_dir, case_name))
        test(case_data_dir, ckp_path, case_save_dir, win_size = win_size)


    