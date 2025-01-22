import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict 
import random

import torch
import torchvision.transforms.functional as F
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset 


from .utils import *
from .constants import n_pathology, pathology_paths, pathology_prob_paths, \
    n_neutral_labels, label_list_segmentation, augmentation_funcs, processing_funcs
import utils.interpol as interpol

from utils.misc import viewVolume


from FluidAnomaly.DiffEqs.pde import AdvDiffPDE
 
seed = int(time.time())
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)

class BaseGen(Dataset):
    """
    BaseGen dataset
    """ 
    def __init__(self, gen_args, dataset_name, setup_dict, device='cpu'):

        self.gen_args = gen_args 
        self.split = gen_args.split 

        self.synth_args = self.gen_args.generator
        self.shape_gen_args = gen_args.pathology_shape_generator
        self.real_image_args = gen_args.real_image_generator
        self.synth_image_args = gen_args.synth_image_generator 
        self.augmentation_steps = vars(gen_args.augmentation_steps)
        self.dataset_name = dataset_name
        self.device = device

        self.input_prob = vars(vars(gen_args.modality_probs)[dataset_name])
        self.mix_synth_prob = gen_args.mix_synth_prob

        self.prepare_paths(setup_dict)
        self.prepare_tasks()
        self.prepare_grid()
        self.prepare_one_hot()


    def __len__(self):
        return len(self.names)


    def prepare_paths(self, setup_dict):

        # load paths
        self.paths = defaultdict(lambda: None) 
        for key, value in setup_dict['paths'].items():
            if value:
                self.paths[key] = os.path.join(setup_dict['root'], value)

        # read names
        split_file = open(os.path.join(setup_dict['root'], setup_dict[self.split]), 'r')
        names = []
        for subj in split_file.readlines():
            names.append(subj.strip())
        self.names = names

        self.pathology_type = setup_dict['pathology_type']
        # save available modalities
        self.modalities = setup_dict['modalities']

    def prepare_tasks(self):
        self.tasks = [key for (key, value) in vars(self.gen_args.task).items() if value]
        if 'bias_fields' in self.tasks and 'segmentation' not in self.tasks:
            # add segmentation mask for computing bias_field_soft_mask
            self.tasks += ['segmentation']
        if ('encode_anomaly' in self.tasks or 'pathology' in self.tasks) and self.synth_args.augment_pathology: # and self.synth_args.random_shape_prob < 1.: 
            self.t = torch.from_numpy(np.arange(self.shape_gen_args.max_nt) * self.shape_gen_args.dt).to(self.device)
            with torch.no_grad():
                self.adv_pde = AdvDiffPDE(data_spacing=[1., 1., 1.], 
                                        perf_pattern='adv', 
                                        V_type='vector_div_free', 
                                        V_dict={},
                                        BC=self.shape_gen_args.bc, 
                                        dt=self.shape_gen_args.dt, 
                                        device=self.device
                                        )
        else:
            self.t, self.adv_pde = None, None
        for task_name in self.tasks:
            if task_name not in processing_funcs.keys(): 
                print('Warning: Function for task "%s" not found' % task_name)


    def prepare_grid(self): 
        self.size = self.synth_args.size

        # Get resolution of training data
        aff = nib.load(os.path.join(self.paths['Gen'], self.names[0])).affine
        self.res_training_data = np.sqrt(np.sum(abs(aff[:-1, :-1]), axis=0))

        xx, yy, zz = np.meshgrid(range(self.size[0]), range(self.size[1]), range(self.size[2]), sparse=False, indexing='ij')
        self.xx = torch.tensor(xx, dtype=torch.float, device=self.device)
        self.yy = torch.tensor(yy, dtype=torch.float, device=self.device)
        self.zz = torch.tensor(zz, dtype=torch.float, device=self.device)
        self.c = torch.tensor((np.array(self.size) - 1) / 2, dtype=torch.float, device=self.device)
        self.xc = self.xx - self.c[0]
        self.yc = self.yy - self.c[1]
        self.zc = self.zz - self.c[2]
        return
    
    def prepare_one_hot(self): 
        # Matrix for one-hot encoding (includes a lookup-table)
        n_labels = len(label_list_segmentation)
        self.lut = torch.zeros(10000, dtype=torch.long, device=self.device)
        for l in range(n_labels):
            self.lut[label_list_segmentation[l]] = l
        self.onehotmatrix = torch.eye(n_labels, dtype=torch.float, device=self.device)
        
        nlat = int((n_labels - n_neutral_labels) / 2.0)
        self.vflip = np.concatenate([np.array(range(n_neutral_labels)),
                                np.array(range(n_neutral_labels + nlat, n_labels)),
                                np.array(range(n_neutral_labels, n_neutral_labels + nlat))])
        return

    
    def random_affine_transform(self, shp):
        if not self.synth_args.augment:
            A = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float, device=self.device)
            scaling_factor_distances = 1.
            c2 = torch.tensor((np.array(shp[0:3]) - 1)/2, dtype=torch.float, device=self.device)
        else: 
            rotations = (2 * self.synth_args.max_rotation * np.random.rand(3) - self.synth_args.max_rotation) / 180.0 * np.pi
            shears = (2 * self.synth_args.max_shear * np.random.rand(3) - self.synth_args.max_shear)
            scalings = 1 + (2 * self.synth_args.max_scaling * np.random.rand(3) - self.synth_args.max_scaling)
            scaling_factor_distances = np.prod(scalings) ** .33333333333 
            A = torch.tensor(make_affine_matrix(rotations, shears, scalings), dtype=torch.float, device=self.device)

            # sample center 
            if self.synth_args.random_shift:
                max_shift = (torch.tensor(np.array(shp[0:3]) - self.size, dtype=torch.float, device=self.device)) / 2
                max_shift[max_shift < 0] = 0
                c2 = torch.tensor((np.array(shp[0:3]) - 1)/2, dtype=torch.float, device=self.device) + (2 * (max_shift * torch.rand(3, dtype=float, device=self.device)) - max_shift)
            else:
                c2 = torch.tensor((np.array(shp[0:3]) - 1)/2, dtype=torch.float, device=self.device)
        
        return scaling_factor_distances, A, c2

    def random_nonlinear_transform(self, photo_mode, spac):
        nonlin_scale = self.synth_args.nonlin_scale_min + np.random.rand(1) * (self.synth_args.nonlin_scale_max - self.synth_args.nonlin_scale_min)
        size_F_small = np.round(nonlin_scale * np.array(self.size)).astype(int).tolist()
        if photo_mode:
            size_F_small[1] = np.round(self.size[1]/spac).astype(int)
        nonlin_std = self.synth_args.nonlin_std_max * np.random.rand()
        Fsmall = nonlin_std * torch.randn([*size_F_small, 3], dtype=torch.float, device=self.device)
        F = myzoom_torch(Fsmall, np.array(self.size) / size_F_small)
        if photo_mode:
            F[:, :, :, 1] = 0

        if 'surface' in self.tasks: # need to integrate the non-linear deformation fields for inverse
            steplength = 1.0 / (2.0 ** self.synth_args.n_steps_svf_integration)
            Fsvf = F * steplength
            for _ in range(self.synth_args.n_steps_svf_integration):
                Fsvf += fast_3D_interp_torch(Fsvf, self.xx + Fsvf[:, :, :, 0], self.yy + Fsvf[:, :, :, 1], self.zz + Fsvf[:, :, :, 2], 'linear')
            Fsvf_neg = -F * steplength
            for _ in range(self.synth_args.n_steps_svf_integration):
                Fsvf_neg += fast_3D_interp_torch(Fsvf_neg, self.xx + Fsvf_neg[:, :, :, 0], self.yy + Fsvf_neg[:, :, :, 1], self.zz + Fsvf_neg[:, :, :, 2], 'linear')
            F = Fsvf
            Fneg = Fsvf_neg
        else:
            Fneg = None
        return F, Fneg
    
    def generate_and_store_deformation(self, setups, shp, aff, flip2orig):

        # generate affine deformation
        scaling_factor_distances, A, c2 = self.random_affine_transform(shp)
        
        # generate nonlinear deformation 
        if self.synth_args.nonlinear_transform:
            F, Fneg = self.random_nonlinear_transform(setups['photo_mode'], setups['spac']) 
        else:
            F, Fneg = None, None

        # deform the image grid 
        xx2, yy2, zz2, x1, y1, z1, x2, y2, z2 = self.deform_grid(shp, A, c2, F)  
        
        return {'orig_shp': shp,
                'scaling_factor_distances': scaling_factor_distances, 
                'A': A, 
                'c2': c2, 
                'F': F, 
                'Fneg': Fneg, 
                'grid': [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2],
                'aff_orig': aff,
                'flip2orig': flip2orig}
    
    def deform_grid(self, shp, A, c2, F): 
        if F is not None:
            # deform the images (we do nonlinear "first" ie after so we can do heavy coronal deformations in photo mode)
            xx1 = self.xc + F[:, :, :, 0]
            yy1 = self.yc + F[:, :, :, 1]
            zz1 = self.zc + F[:, :, :, 2]
        else:
            xx1 = self.xc
            yy1 = self.yc
            zz1 = self.zc
 
        xx2 = A[0, 0] * xx1 + A[0, 1] * yy1 + A[0, 2] * zz1 + c2[0]
        yy2 = A[1, 0] * xx1 + A[1, 1] * yy1 + A[1, 2] * zz1 + c2[1]
        zz2 = A[2, 0] * xx1 + A[2, 1] * yy1 + A[2, 2] * zz1 + c2[2]  
        xx2[xx2 < 0] = 0
        yy2[yy2 < 0] = 0
        zz2[zz2 < 0] = 0
        xx2[xx2 > (shp[0] - 1)] = shp[0] - 1
        yy2[yy2 > (shp[1] - 1)] = shp[1] - 1
        zz2[zz2 > (shp[2] - 1)] = shp[2] - 1

        # Get the margins for reading images
        x1 = torch.floor(torch.min(xx2))
        y1 = torch.floor(torch.min(yy2))
        z1 = torch.floor(torch.min(zz2))
        x2 = 1+torch.ceil(torch.max(xx2))
        y2 = 1 + torch.ceil(torch.max(yy2))
        z2 = 1 + torch.ceil(torch.max(zz2))
        xx2 -= x1
        yy2 -= y1
        zz2 -= z1

        x1 = x1.cpu().numpy().astype(int)
        y1 = y1.cpu().numpy().astype(int)
        z1 = z1.cpu().numpy().astype(int)
        x2 = x2.cpu().numpy().astype(int)
        y2 = y2.cpu().numpy().astype(int)
        z2 = z2.cpu().numpy().astype(int)

        return xx2, yy2, zz2, x1, y1, z1, x2, y2, z2

    def deform_flip2orig(self, I_flip, deform_dict):
        # deform flip to non-flip space
        M = torch.linalg.inv(deform_dict['aff_orig']) @ deform_dict['flip2orig'] @ deform_dict['aff_orig']
        xx2 = M[0,0] * self.xx + M[0,1] * self.yy + M[0,2] * self.zz + M[0,3]
        yy2 = M[1,0] * self.xx + M[1,1] * self.yy + M[1,2] * self.zz + M[1,3]
        zz2 = M[2,0] * self.xx + M[2,1] * self.yy + M[2,2] * self.zz + M[2,3] 
        I_flip2orig = fast_3D_interp_torch(I_flip, xx2, yy2, zz2, 'linear')
        return I_flip2orig

    def augment_sample(self, name, I, I_flip2orig, setups, deform_dict, res, target, pathol_direction = None, input_mode = 'synth'):

        [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2] = deform_dict['grid']

        if not isinstance(I, torch.Tensor): # real image mode
            I = torch.squeeze(torch.tensor(I.get_fdata()[x1:x2, y1:y2, z1:z2].astype(float), dtype=torch.float, device=self.device))
            
            if self.pathology_type is None and 'pathology' in target and isinstance(target['pathology'], torch.Tensor): # healthy dataset + synth pathol 
                target['pathology'][0][I < 1e-3] = 0
                target['pathology_prob'][0][I < 1e-3] = 0  
                # encode pathology 
                #viewVolume(I, names = ['before'], save_dir = '~/results/tmp')
                #viewVolume(target['pathology'], names = ['pathol'], save_dir = '~/results/tmp')
                #viewVolume(target['pathology_prob'], names = ['pathol_prob'], save_dir = '~/results/tmp')
                I = self.encode_pathology(I, target['pathology'], target['pathology_prob'], self.get_pathology_direction(input_mode)) 
                I[I < 0.] = 0.
                #viewVolume(I, names = ['after'], save_dir = '~/results/tmp')

            # deform flip to non-flip space
            I_flip2orig = self.deform_flip2orig(torch.flip(I, [0]), deform_dict)

            # Deform grid
            I_def = fast_3D_interp_torch(I, xx2, yy2, zz2, 'linear')
            I_flip2orig_def = fast_3D_interp_torch(I_flip2orig, xx2, yy2, zz2, 'linear')

        else: # synth mode
            I_def = fast_3D_interp_torch(I, xx2, yy2, zz2) 
            I_flip2orig_def = fast_3D_interp_torch(I_flip2orig, xx2, yy2, zz2)


        if input_mode == 'CT':
            I_def = torch.clamp(I_def, min = 0., max = 80.)
            I_flip2orig_def = torch.clamp(I_flip2orig_def, min = 0., max = 80.)

        # prepare input and target
        if 'super_resolution' in self.tasks:
            sample.update({'orig': torch.flip(I_def, [0])[None] if setups['flip'] else I_def[None]})
            
        # Augment sample
        aux_dict = {}
        augmentation_steps = self.augmentation_steps['synth'] if input_mode == 'synth' else self.augmentation_steps['real']
        for func_name in augmentation_steps:
            I_def, I_flip2orig_def, aux_dict = augmentation_funcs[func_name](I = I_def, I_flip = I_flip2orig_def, aux_dict = aux_dict, cfg = self.gen_args.generator, 
                                                         input_mode = input_mode, setups = setups, size = self.size, res = res, device = self.device)

        # Back to original resolution 
        if self.synth_args.bspline_zooming:
            I_def = interpol.resize(I_def, shape=self.size, anchor='edge', interpolation=3, bound='dct2', prefilter=True) 
            I_flip2orig_def = interpol.resize(I_flip2orig_def, shape=self.size, anchor='edge', interpolation=3, bound='dct2', prefilter=True) 
        else:
            I_def = myzoom_torch(I_def, 1 / aux_dict['factors']) 
            I_flip2orig_def = myzoom_torch(I_flip2orig_def, 1 / aux_dict['factors']) 

        I_def[I_def < 0.] = 0.
        I_flip2orig_def[I_flip2orig_def < 0.] = 0.
        I_final = I_def / torch.max(I_def)
        I_flip2orig_final = I_flip2orig_def / torch.max(I_flip2orig_def)
        
        sample = {'input': torch.flip(I_final, [0])[None] if setups['flip'] else I_final[None],
                  'input_flip': torch.flip(I_flip2orig_final, [0])[None] if setups['flip'] else I_flip2orig_final[None]}
        if 'bias_fields' in self.tasks:
            sample.update({'bias_field_log': torch.flip(aux_dict['BFlog'], [0])[None] if setups['flip'] else aux_dict['BFlog'][None]})

        return sample, target
    

    def generate_sample(self, name, G, setups, deform_dict, res, target):  
        
        [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2] = deform_dict['grid']

        # Generate contrasts
        mus, sigmas = self.get_contrast(setups['photo_mode'])

        G = torch.squeeze(torch.tensor(G.get_fdata()[x1:x2, y1:y2, z1:z2].astype(float), dtype=torch.float, device=self.device))
        G[G > 255] = 0 # kill extracerebral regions
        G[G == 77] = 2 # merge WM lesion to white matter region
        Gr = torch.round(G).long()
        
        SYN = mus[Gr] + sigmas[Gr] * torch.randn(Gr.shape, dtype=torch.float, device=self.device)

        if self.synth_args.pv:
            mask = (G!=Gr)
            SYN[mask] = 0
            Gv = G[mask]
            isv = torch.zeros(Gv.shape, dtype=torch.float, device=self.device )
            pw = (Gv<=3) * (3-Gv)
            isv += pw * mus[2] + pw * sigmas[2] * torch.randn(Gv.shape, dtype=torch.float, device=self.device)
            pg = (Gv<=3) * (Gv-2) + (Gv>3) * (4-Gv)
            isv += pg * mus[3] + pg * sigmas[3] * torch.randn(Gv.shape, dtype=torch.float, device=self.device)
            pcsf = (Gv>=3) * (Gv-3)
            isv += pcsf * mus[4] + pcsf * sigmas[4] * torch.randn(Gv.shape, dtype=torch.float, device=self.device)
            SYN[mask] = isv
        SYN[SYN < 0] = 0
            
        if 'pathology' in target and isinstance(target['pathology'], torch.Tensor) and target['pathology'].sum() > 0:
            wm_mask = (Gr==2) | (Gr==41)
            wm_mean = (SYN * wm_mask).sum() / wm_mask.sum()  
            gm_mask = (Gr!=0) & (Gr!=2) & (Gr!=41)
            gm_mean = (SYN * gm_mask).sum() / gm_mask.sum() 

            #print('bef', target['pathology'].sum())
            #print(target['pathology'].shape, SYN.shape, G.shape)
            target['pathology'][0][G < 1] = 0
            target['pathology_prob'][0][G < 1] = 0 
            #print('aft', target['pathology'].sum())

            # determine to be T1-resembled or T2-resembled
            #if pathol_direction: lesion should be brigher than WM.mean() 
            # pathol_direction: +1: T2-like; -1: T1-like
            pathol_direction = self.get_pathology_direction('synth', gm_mean > wm_mean)

            # encode pathology
            #viewVolume(SYN, names = ['before'], save_dir = '~/results/tmp')
            #viewVolume(target['pathology'], names = ['pathol'], save_dir = '~/results/tmp')
            #viewVolume(target['pathology_prob'], names = ['pathol_prob'], save_dir = '~/results/tmp')
            SYN = self.encode_pathology(SYN, target['pathology'], target['pathology_prob'], pathol_direction)
            #viewVolume(SYN, names = ['after'], save_dir = '~/results/tmp')
            SYN[SYN < 0.] = 0.
        else:
            pathol_direction = None
            target['pathology'] = 0.
            target['pathology_prob'] = 0. 

        SYN[SYN < 0.] = 0.

        # prepare flipped conditional input
        SYN_flip = torch.flip(SYN, [0])
        # deform flip to non-flip space
        SYN_flip = self.deform_flip2orig(SYN_flip, deform_dict)


        #SYN = fast_3D_interp_torch(SYN, xx2, yy2, zz2) 
        #SYN_flip = fast_3D_interp_torch(SYN_flip, xx2, yy2, zz2) 

        # Make random linear combinations TODO add before pathology encoding and deformation
        #if np.random.rand() < self.gen_args.mix_synth_prob: 
        #    v = torch.rand(4)
        #    v[2] = 0 if 'T2' not in target else v[2]
        #    v[3] = 0 if 'FLAIR' not in target else v[3]
        #    v /= torch.sum(v) 
        #    if 'T1' in target:
        #        SYN = v[0] * SYN + v[1] * target['T1'][0]
        #    if 'T2' in target:
        #        SYN += v[2] * target['T2'][0]
        #    if 'FLAIR' in target:
        #        SYN += v[3] * target['FLAIR'][0] 
            
        return target['pathology'], target['pathology_prob'], self.augment_sample(name, SYN, SYN_flip, setups, deform_dict, res, target, pathol_direction = pathol_direction)
    
    def get_pathology_direction(self, input_mode, pathol_direction = None):  
        #if np.random.rand() < 0.1: # in some (rare) cases, randomly pick the direction
        #    return random.choice([True, False])
        
        if pathol_direction is not None: # for synth image
            return pathol_direction
        
        if input_mode in ['T1', 'CT']:
            return False
        
        if input_mode in ['T2', 'FLAIR']:
            return True
        
        #return random.choice([True, False])


    def get_contrast(self, photo_mode):
        # Sample Gaussian image
        mus = 25 + 200 * torch.rand(10000, dtype=torch.float, device=self.device)
        sigmas = 5 + 20 * torch.rand(10000, dtype=torch.float, device=self.device)

        if np.random.rand() < self.synth_args.ct_prob:
            darker = 25 + 10 * torch.rand(1, dtype=torch.float, device=self.device)[0]
            for l in ct_brightness_group['darker']:
                mus[l] = darker
            dark = 90 + 20 * torch.rand(1, dtype=torch.float, device=self.device)[0]
            for l in ct_brightness_group['dark']:
                mus[l] = dark
            bright = 110 + 20 * torch.rand(1, dtype=torch.float, device=self.device)[0]
            for l in ct_brightness_group['bright']:
                mus[l] = bright
            brighter = 150 + 50 * torch.rand(1, dtype=torch.float, device=self.device)[0]
            for l in ct_brightness_group['brighter']:
                mus[l] = brighter
        if photo_mode or np.random.rand(1)<0.5: # set the background to zero every once in a while (or always in photo mode)
            mus[0] = 0
        return mus, sigmas
    
    def get_random_params(self): 
        photo_mode = np.random.rand() < self.synth_args.photo_prob
        pathol_mode = np.random.rand() < self.synth_args.pathology_prob
        pathol_random_shape = np.random.rand() < self.synth_args.random_shape_prob
        spac = 2.0 + 10 * np.random.rand() if photo_mode else None 
        flip = np.random.randn() < self.synth_args.flip_prob
        
        if not self.synth_args.augment:
            photo_mode, spac, flip = False, None, False
            resolution = np.array([1.0, 1.0, 1.0])
            thickness = np.array([1.0, 1.0, 1.0])
        elif photo_mode: 
            resolution = np.array([self.res_training_data[0], spac, self.res_training_data[2]])
            thickness = np.array([self.res_training_data[0], 0.0001, self.res_training_data[2]])
        else:
            resolution, thickness = resolution_sampler()
        return {'resolution': resolution, 'thickness': thickness, 
                'photo_mode': photo_mode, 'pathol_mode': pathol_mode, 
                'pathol_random_shape': pathol_random_shape,
                'spac': spac, 'flip': flip}
    
    
    def encode_pathology(self, I, P, Pprob, pathol_direction = None):

        if pathol_direction is None: # True: T2/FLAIR-resembled, False: T1-resembled
            pathol_direction = random.choice([True, False])

        P, Pprob = torch.squeeze(P), torch.squeeze(Pprob)
        I_mu = (I * P).sum() / P.sum()

        p_mask = torch.round(P).long()
        #pth_mus = I_mu/4 + I_mu/2 * torch.rand(10000, dtype=torch.float, device=self.device)
        pth_mus = I_mu + I_mu * torch.rand(10000, dtype=torch.float, device=self.device) # enforce the pathology pattern harder!
        pth_mus = pth_mus if pathol_direction else -pth_mus 
        pth_sigmas = I_mu/4 * torch.rand(10000, dtype=torch.float, device=self.device)
        I += Pprob * (pth_mus[p_mask] + pth_sigmas[p_mask] * torch.randn(p_mask.shape, dtype=torch.float, device=self.device))
        I[I < 0] = 0

        #print('encode', P.shape, P.mean(), Pprob.max()) 
        #print('pre', I_mu) 
        #I_mu = (I * P).sum() / P.sum()
        #print('post', I_mu)

        return I

    def read_input(self, name):
        """
        determine input type according to prob (in generator/constants.py)
        Logic: if np.random.rand() < real_image_prob and is real_image_exist --> input real images; otherwise, synthesize images. 
        """
        self.pathol_mode = None # reset per subject
        prob = np.random.rand() 
        if prob < self.input_prob['T1'] and 'T1' in self.modalities and os.path.isfile(os.path.join(self.paths['T1'], name)):
            input_mode = 'T1'
            img, aff, res = read_image(os.path.join(self.paths['T1'], name)) 
        elif prob < self.input_prob['T2'] and 'T2' in self.modalities and os.path.isfile(os.path.join(self.paths['T2'], name)):
            input_mode = 'T2'
            img, aff, res = read_image(os.path.join(self.paths['T2'], name)) 
        elif prob < self.input_prob['FLAIR'] and 'FLAIR' in self.modalities and os.path.isfile(os.path.join(self.paths['FLAIR'], name)):
            input_mode = 'FLAIR'
            img, aff, res = read_image(os.path.join(self.paths['FLAIR'], name)) 
        elif prob < self.input_prob['CT'] and 'CT' in self.modalities and os.path.isfile(os.path.join(self.paths['CT'], name)):
            input_mode = 'CT'
            img, aff, res = read_image(os.path.join(self.paths['CT'], name)) 
        else:
            input_mode = 'synth' 
            img, aff, res = read_image(os.path.join(self.paths['Gen'], name)) 

        if 'flip2orig' in self.paths:
            flip2orig = np.loadtxt(os.path.join(self.paths['flip2orig'], name.split('.nii')[0] + '.txt'), delimiter=' ')
            flip2orig = torch.from_numpy(flip2orig).float().to(self.device) # 4x4
        else:
            flip2orig = None
        
        aff = torch.tensor(aff, dtype=torch.float, device=self.device)

        return input_mode, img, aff, res, flip2orig
    

    def read_and_deform_target(self, idx, exist_keys, task_name, input_mode, setups, deform_dict):
        current_target = {}
        p_prob_path, augment, thres = None, False, 0.1

        if task_name == 'encode_anomaly':
            # NOTE: for now - encode pathology only for healthy cases
            # TODO: what to do if the case has pathology already?
            if self.pathology_type is None: # healthy
                if setups['pathol_mode']: # and input_mode == 'synth':
                    self.pathol_mode = 'synth'
                    if setups['pathol_random_shape']:
                        p_prob_path = 'random_shape'
                        augment, thres = False, self.shape_gen_args.pathol_thres # TODO
                        #augment, thres = True, self.shape_gen_args.pathol_thres 
                    else:
                        p_prob_path = random.choice(pathology_prob_paths)
                        #p_prob_path = '~/data/isles2022_crop/pathology_probability/sub-strokecase0051.nii.gz'
                        if self.gen_args.save_orig_for_visualize:
                            print('Using real pathol', p_prob_path)
                        current_target['p_prob_path'] = p_prob_path
                        augment, thres = self.synth_args.augment_pathology, self.shape_gen_args.pathol_thres 
            else:
                self.pathol_mode = 'real'
                p_prob_path = os.path.join(self.paths['pathology_prob'], self.names[idx])
                augment, thres = False, 1e-7 # use the GT pathology without augmentation
                #print(' add real pathology without augmentation')

            current_target = processing_funcs[task_name](exist_keys, task_name, p_prob_path, setups, deform_dict, self.device,
                                                         pde_augment = augment, 
                                                         pde_func = self.adv_pde, 
                                                         t = self.t, 
                                                         shape_gen_args = self.shape_gen_args, 
                                                         thres = thres,
                                                         save_orig_for_visualize = self.gen_args.save_orig_for_visualize)
            
        elif task_name != 'pathology':
            if task_name in self.paths and os.path.isfile(os.path.join(self.paths[task_name], self.names[idx])):
                current_target = processing_funcs[task_name](exist_keys, task_name, 
                                                            os.path.join(self.paths[task_name], self.names[idx]), 
                                                            setups, deform_dict, self.device, 
                                                            cfg = self.gen_args, 
                                                            onehotmatrix = self.onehotmatrix, lut = self.lut, vflip = self.vflip)
            else:
                current_target = {task_name: 0.}
        return current_target
    
        
    def update_gen_args(self, new_args):
        for key, value in vars(new_args).items():
            vars(self.gen_args.generator)[key] = value 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  

        # read input: real or synthesized image, according to customized prob
        input_mode, img, aff, res, flip2orig = self.read_input(self.names[idx])

        # generate random values
        setups = self.get_random_params()

        # sample and store random deformation
        deform_dict = self.generate_and_store_deformation(setups, img.shape, aff, flip2orig)

        # read and deform target according to the assigned tasks
        target = defaultdict(lambda: None)
        for task_name in self.tasks:
            if task_name in processing_funcs.keys(): 
                target.update(self.read_and_deform_target(idx, target.keys(), task_name, input_mode, setups, deform_dict))
    

        # process or generate input sample
        if input_mode == 'synth':
            self.update_gen_args(self.synth_image_args)
            target['pathology'], target['pathology_prob'], (sample, target) = \
                self.generate_sample(self.names[idx], img, setups, deform_dict, res, target)
        else:
            self.update_gen_args(self.real_image_args) # milder noise injection for real images
            sample, target = self.augment_sample(self.names[idx], img, None, setups, deform_dict, res, target, 
                                            pathol_direction = self.get_pathology_direction(input_mode), input_mode = input_mode)
        
        # prepare and deform pathology
        if 'pathology' in target and isinstance(target['pathology'], torch.Tensor):
            [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2] = deform_dict['grid']
            # compute pathology_flip2orig
            target['pathology_flip_prob'] = self.deform_flip2orig(torch.flip(target['pathology_prob'][0], [0]), deform_dict)[None]
            # deform pathology to aug_I space
            target['pathology_prob'] = fast_3D_interp_torch(target['pathology_prob'][0], xx2, yy2, zz2)[None]
            target['pathology'] = binarize(target['pathology_prob'], thres = self.shape_gen_args.pathol_thres)
            target['pathology_flip_prob'] = fast_3D_interp_torch(target['pathology_flip_prob'][0], xx2, yy2, zz2)[None]
            target['pathology_flip'] = binarize(target['pathology_flip_prob'], thres = self.shape_gen_args.pathol_thres)

            if setups['flip']:
                target['pathology'], target['pathology_prob'] = torch.flip(target['pathology'], [1]), torch.flip(target['pathology_prob'], [1]) 
                target['pathology_flip'], target['pathology_flip_prob'] = torch.flip(target['pathology_flip'], [1]), torch.flip(target['pathology_flip_prob'], [1]) 
            target['common_pathology'] = target['pathology'] * target['pathology_flip']
            
        return input_mode, self.pathol_mode, target, sample




# An example of customized dataset from BaseSynth
class UNAGen(BaseGen):
    """
    UNAGen dataset
    UNAGen enables intra-subject augmentation, i.e., each subject will have multiple augmentations
    """
    def __init__(self, gen_args, dataset_name, setup_dict, device='cpu'):  
        super(UNAGen, self).__init__(gen_args, dataset_name, setup_dict, device)

        self.all_samples = gen_args.generator.all_samples 
        self.mild_samples = gen_args.generator.mild_samples 
        self.mild_generator_args = gen_args.mild_generator
        self.severe_generator_args = gen_args.severe_generator
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  

        # read input: real or synthesized image, according to customized prob
        input_mode, img, aff, res, flip2orig = self.read_input(self.names[idx])

        # generate random values
        setups = self.get_random_params()
 
        # sample and store random deformation
        deform_dict = self.generate_and_store_deformation(setups, img.shape, aff, flip2orig)

        # read and deform target according to the assigned tasks
        target = defaultdict(lambda: 1.)
        target['name'] = self.names[idx]
        for task_name in self.tasks:
            if task_name in processing_funcs.keys(): 
                target.update(self.read_and_deform_target(idx, target.keys(), task_name, input_mode, setups, deform_dict)) 

        # process or generate intra-subject input samples 
        samples = []
        for i_sample in range(self.all_samples):
            if i_sample < self.mild_samples:  
                self.update_gen_args(self.mild_generator_args)
                if input_mode == 'synth':
                    self.update_gen_args(self.synth_image_args)
                    target['pathology'], target['pathology_prob'], (sample, target) = \
                        self.generate_sample(self.names[idx], img, setups, deform_dict, res, target)
                else:
                    self.update_gen_args(self.real_image_args)
                    sample, target = self.augment_sample(self.names[idx], img, None, setups, deform_dict, res, target, 
                                                 pathol_direction = self.get_pathology_direction(input_mode), input_mode = input_mode)
            else: 
                self.update_gen_args(self.severe_generator_args)
                if input_mode == 'synth':
                    self.update_gen_args(self.synth_image_args)
                    target['pathology'], target['pathology_prob'], (sample, target) = \
                        self.generate_sample(self.names[idx], img, setups, deform_dict, res, target) 
                else:
                    self.update_gen_args(self.real_image_args)
                    sample, target = self.augment_sample(self.names[idx], img, None, setups, deform_dict, res, target, 
                                                 pathol_direction = self.get_pathology_direction(input_mode), input_mode = input_mode)

            samples.append(sample)
        
        # prepare and deform pathology
        if 'pathology' in target and isinstance(target['pathology'], torch.Tensor):
            [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2] = deform_dict['grid']
            # compute pathology_flip2orig
            target['pathology_flip_prob'] = self.deform_flip2orig(torch.flip(target['pathology_prob'][0], [0]), deform_dict)[None]
            # deform pathology to aug_I space
            target['pathology_prob'] = fast_3D_interp_torch(target['pathology_prob'][0], xx2, yy2, zz2)[None]
            target['pathology'] = binarize(target['pathology_prob'], thres = self.shape_gen_args.pathol_thres)
            target['pathology_flip_prob'] = fast_3D_interp_torch(target['pathology_flip_prob'][0], xx2, yy2, zz2)[None]
            target['pathology_flip'] = binarize(target['pathology_flip_prob'], thres = self.shape_gen_args.pathol_thres)

            if setups['flip']:
                target['pathology'], target['pathology_prob'] = torch.flip(target['pathology'], [1]), torch.flip(target['pathology_prob'], [1]) 
                target['pathology_flip'], target['pathology_flip_prob'] = torch.flip(target['pathology_flip'], [1]), torch.flip(target['pathology_flip_prob'], [1]) 
            target['common_pathology'] = target['pathology'] * target['pathology_flip']
            


        return input_mode, self.pathol_mode, target, samples