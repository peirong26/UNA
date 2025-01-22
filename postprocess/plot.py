import os, sys, warnings, shutil, time, datetime
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 

import numpy as np 


from utils.misc import make_dir, viewVolume, MRIread
import utils.test_utils as utils


##############################################
##############################################
##############################################

import os
import numpy as np  
from math import ceil
import torch
import torch.nn.functional as F
from PIL import Image
from collections import defaultdict

from utils.misc import make_dir


def match_shape(array, shape):
    # array: (channel_dim, *orig_shape)
    array = array[None]
    if list(array.shape[2:]) != list(shape):
        array = F.interpolate(array, size=shape) 
    return array[0]

def pad_shape(array_list):
    max_shape = [0] * len(array_list[0].shape)

    for array in array_list:
        max_shape = [max(max_shape[dim], array.shape[dim]) for dim in range(len(max_shape))]  
    pad_array_list = []
    for array in array_list: 
        start = [(max_shape[dim] - array.shape[dim]) // 2 for dim in range(len(max_shape))] 
        if len(start) == 2:
            pad_array = np.zeros((max_shape[0], max_shape[1]))
            pad_array[start[0] : start[0] + array.shape[0], start[1] : start[1] + array.shape[1]] = array
        elif len(start) == 3:
            pad_array = np.zeros((max_shape[0], max_shape[1], max_shape[2]))
            pad_array[start[0] : start[0] + array.shape[0], start[1] : start[1] + array.shape[1], start[2] : start[2] + array.shape[2]] = array
        elif len(start) == 4:
            pad_array = np.zeros((max_shape[0], max_shape[1], max_shape[2], max_shape[3]))
            pad_array[start[0] : start[0] + array.shape[0], start[1] : start[1] + array.shape[1], start[2] : start[2] + array.shape[2], start[3] : start[3] + array.shape[3]] = array
        
        pad_array_list.append(pad_array) 
    return pad_array_list


def even_sample(orig_len, num):
     idx = []
     length = float(orig_len)
     for i in range(num):
             idx.append(int(ceil(i * length / num)))
     return idx


def normalize(nda, channel = None):
    if channel is not None:
        nda_max = np.max(nda, axis = channel, keepdims = True)
        nda_min = np.min(nda, axis = channel, keepdims = True)
    else:
        nda_max = np.max(nda)
        nda_min = np.min(nda)
    return (nda - nda_min) / (nda_max - nda_min + 1e-7)






class ImageVisualizer(object):

    def __init__(self): 
        self.draw_border = False
        self.vis_spacing = [8, 8, 8] 

    def create_image_row(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=1)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            out.append(normalize(self.create_image_row(arg))) 
        return np.concatenate(out, axis=0) 

    def prepare_for_itk(self, array): # (s, r, c, *)
        return array[:, ::-1, :]

    def prepare_for_png(self, array, normalize = False): # (s, r, c, *) 
        slc = array[::self.vis_spacing[0]] # (s', r, c *)
        row = array[:, ::self.vis_spacing[1]].transpose((1, 0, 2, 3))[:, ::-1] # (s, r', c, *) -> (r', s, c, *)
        col = array[:, :, ::self.vis_spacing[2]].transpose((2, 0, 1, 3))[:, ::-1] # (s, r, c', *) -> (c', s, r, *)

        if normalize:
            slc = (slc - np.min(slc)) / (np.max(slc) - np.min(slc))
            row = (slc - np.min(slc)) / (np.max(slc) - np.min(row))
            col = (slc - np.min(slc)) / (np.max(slc) - np.min(col))
        return slc, row, col


    def visualize_sample(self, name, input, out_dir, postfix = ''): # (s, r, c)

        slc_images, row_images, col_images = [], [], []
        
        input_slc, input_row, input_col = self.prepare_for_png(input, normalize = False)
        slc_images.append(input_slc)
        row_images.append(input_row)
        col_images.append(input_col)

        # add row gap 
        #gap = [np.zeros_like(slc_images[0])]
        #all_images = slc_images + gap + row_images + gap + col_images
        all_images = slc_images + row_images + col_images
        all_images = pad_shape(all_images)
        all_image = self.create_image_grid(*all_images)
        all_image = (255 * all_image).astype(np.uint8)  
        Image.fromarray(all_image[:, :, 0]).save(os.path.join(out_dir, name + postfix + '.png')) # grey scale image last channel == 1
        return 
    
    
    def visualize_samples(self, name, inputs, out_dir, postfix = ''): # list of (s, r, c)
        
        slc_images, row_images, col_images = [], [], []


        for input in inputs:
        
            input_slc, input_row, input_col = self.prepare_for_png(input, normalize = False)
            slc_images.append(input_slc)
            row_images.append(input_row)
            col_images.append(input_col) 

        # add row gap 
        gap = [np.zeros_like(slc_images[0])]
        all_images = slc_images + gap + row_images + gap + col_images
        all_images = slc_images + row_images + col_images
        all_images = pad_shape(all_images)
        all_image = self.create_image_grid(*all_images)
        all_image = (255 * all_image).astype(np.uint8)  
        Image.fromarray(all_image[:, :, 0]).save(os.path.join(out_dir, name + postfix + '.png')) # grey scale image last channel == 1
        return 








##############################################
##############################################
##############################################

visualizer = ImageVisualizer()


visual_option = 'demo' # demo, dataset, result


if visual_option == 'demo': 

    #main_dir = '~/results/demo_synth/'
    #save_dir = make_dir('~/results/plot_demo_synth', reset = False)
    main_dir = '~/results/demo_synth_random_shape/'
    save_dir = make_dir('~/results/plot_demo_synth_random_shape', reset = False)


    subj_names = os.listdir(main_dir)

    for i_subj, subj_name in enumerate(subj_names):
        print('  Processing %s (%d/%d)' % (subj_name, i_subj, len(subj_names)))
        subj_dir = os.path.join(main_dir, subj_name)

        prefix = '%s.%s' %  (str(i_subj).zfill(3), subj_name.split('.nii')[0])

        file_names = os.listdir(subj_dir)

        pathol0_path = None
        pathol9_path = None
        img_path = None
        for file_name in file_names:
            if file_name.endswith('diseased.nii.gz'):
                img_path = os.path.join(subj_dir, file_name)
            elif file_name == 'pathology_progress_0.nii.gz': 
                pathol0_path = os.path.join(subj_dir, file_name)
            elif file_name == 'pathology_progress_9.nii.gz': 
                pathol9_path = os.path.join(subj_dir, file_name)

        assert pathol0_path is not None and pathol9_path is not None and img_path is not None

        img = MRIread(img_path, im_only = True)
        img = np.nan_to_num(np.squeeze(img))[..., None] 

        pathol0 = MRIread(pathol0_path, im_only = True)
        pathol0 = np.nan_to_num(np.squeeze(pathol0))[..., None] 

        pathol9 = MRIread(pathol9_path, im_only = True)
        pathol9 = np.nan_to_num(np.squeeze(pathol9))[..., None] 
        
        visualizer.visualize_samples(prefix, [pathol0, pathol9, img], save_dir)

        try:
            visualizer.visualize_samples(prefix, [pathol0, pathol9, img], save_dir)
        except:
            print('  Ploting exception raised, skipping...')


elif visual_option == 'dataset':

    #main_dir = '~/data/synth_anomaly'
    #save_dir = make_dir('~/results/plot_synth_anomaly', reset = False)
    main_dir = '~/data/synth_anomaly_synth'
    save_dir = make_dir('~/results/plot_synth_anomaly_synth', reset = False) 


    dataset_names = ['orig-T1', 'orig-T2', 'orig-FLAIR', 'orig-CT']  

    for i, dataset_name in enumerate(dataset_names):
        print('Now ploting %s (%d/%d)' % (dataset_name, i+1, len(dataset_names)))

        curr_save_dir = make_dir(os.path.join(save_dir, dataset_name), reset = True)


        img_dir = os.path.join(main_dir, dataset_name, 'synth')
        pathol_dir = os.path.join(main_dir, dataset_name, 'pathology_maps_segmentation')

        subj_names = os.listdir(img_dir)

        for i_subj, subj_name in enumerate(subj_names):
            print('  Processing %s (%d/%d)' % (subj_name, i_subj+1, len(subj_names)))

            prefix = '%s.%s' %  (str(i_subj).zfill(3), subj_name.split('.nii')[0])

            pathol = MRIread(os.path.join(pathol_dir, subj_name), im_only = True)
            pathol = np.nan_to_num(np.squeeze(pathol))[..., None] 

            synth = MRIread(os.path.join(img_dir, subj_name), im_only = True)
            synth = np.nan_to_num(np.squeeze(synth))[..., None]

            try:
                #visualizer.visualize_sample(prefix + '-pathol', pathol, curr_save_dir)
                #visualizer.visualize_sample(prefix + '-synth', synth, curr_save_dir)
                visualizer.visualize_samples(prefix, [synth, pathol], curr_save_dir)
            except:
                print('  Ploting exception raised, skipping...')


elif visual_option == 'result':

    #main_dir = '~/results/test'
    #save_dir = make_dir('~/results/plot_test', reset = False)  
    main_dir = '~/results/test_synth'
    save_dir = make_dir('~/results/plot_test_synth', reset = True) 
    
    model_names = os.listdir(main_dir)

    for i_model, model_name in enumerate(model_names):
        print('Now ploting model: %s (%d/%d)' % (model_name, i_model+1, len(model_names)))
        model_save_dir = make_dir(os.path.join(save_dir, model_name), reset = False)

        model_dir = os.path.join(main_dir, model_name)

        dataset_names = os.listdir(model_dir) 
        for i_dataset, dataset_name in enumerate(dataset_names):

            #if dataset_name != 'ISLES':
            #    continue


            print('    Processing dataset %s (%d/%d)' % (dataset_name, i_dataset+1, len(dataset_names)))
            dataset_dir = os.path.join(model_dir, dataset_name)
            dataset_save_dir = make_dir(os.path.join(model_save_dir, dataset_name), reset = True)

            subj_names = os.listdir(dataset_dir)  
            for i_subj, subj_name in enumerate(subj_names):
                if not os.path.isdir(os.path.join(dataset_dir, subj_name)):
                    continue
                print('      Processing %s (%d/%d)' % (subj_name, i_subj+1, len(subj_names)))
                subj_dir = os.path.join(dataset_dir, subj_name)

                for modality in ['T1', 'T2', 'FLAIR', 'CT', 'synth']:


                    if os.path.isdir(os.path.join(subj_dir, 'input_'+ modality)):
                        inputs = []

                        prefix = '%s.%s.%s' %  (str(i_subj).zfill(3), subj_name, modality)


                        input_path = os.path.join(subj_dir, subj_name + '.%s.input.nii.gz' % modality)
                        input = MRIread(input_path, im_only = True)
                        input = np.nan_to_num(np.squeeze(input))[..., None] 
                        inputs.append(input)

                        try:
                            pathol_path = os.path.join(subj_dir, subj_name + '.pathol.nii.gz')
                            pathol = MRIread(pathol_path, im_only = True)
                            pathol = np.nan_to_num(np.squeeze(pathol))[..., None] 
                            inputs.append(pathol)
                        except:
                            pass

                        out_t1_path = os.path.join(subj_dir, 'input_%s/out_T1.nii.gz' % modality)
                        out_t1 = MRIread(out_t1_path, im_only = True)
                        out_t1 = np.nan_to_num(np.squeeze(out_t1))[..., None] 
                        inputs.append(out_t1)

                        try:
                            out_flair_path = os.path.join(subj_dir, 'input_%s/out_FLAIR.nii.gz' % modality)
                            out_flair = MRIread(out_flair_path, im_only = True)
                            out_flair = np.nan_to_num(np.squeeze(out_flair))[..., None] 
                            inputs.append(out_flair)
                        except:
                            pass
                

                        # ploting all
                        try:
                            visualizer.visualize_samples(prefix, inputs, dataset_save_dir)
                        except:
                            print('  Ploting exception raised, skipping...')

else:
    raise NotImplementedError



