import os, glob

from .utils import *

augmentation_funcs = {
    'gamma': add_gamma_transform,
    'bias_field': add_bias_field,
    'resample': resample_resolution,
    'noise': add_noise,
}

processing_funcs = {
    'T1': read_and_deform_image,
    'T2': read_and_deform_image,
    'FLAIR': read_and_deform_image,
    'CT': read_and_deform_CT,
    'segmentation': read_and_deform_segmentation,
    'surface': read_and_deform_surface,
    'distance': read_and_deform_distance,
    'bias_fields': read_and_deform_bias_field,
    'registration': read_and_deform_registration,
    'encode_anomaly': read_pathology, #_and_deform_pathology, 
}


dataset_setups = { 

    'ADHD': { 
        'root': '~/data/adhd200_crop',
        'pathology_type': None,
        'train': 'train.txt',
        'test': 'test.txt',
        'modalities': ['T1'],

        'paths':{
                # for synth
                'Gen': 'label_maps_generation', 
                'Dmaps': None, 
                'DmapsBag': None, 

                # real images
                'T1': 'T1', 
                'T2': None, 
                'FLAIR': None,
                'CT': None,

                # flip_to_orig (rigid) deformation field
                'flip2orig': 'T1_flip2orig_synthsr_reg',

                # processed ground truths 
                'surface': None, #'surfaces',  TODO
                'distance': None,  
                'segmentation': 'label_maps_segmentation',
                'bias_fields': None,
                'pathology': None,
                'pathology_prob': None,
        }
    },

    'HCP': { 
        'root': '~/data/hcp_crop',
        'pathology_type': None,
        'train': 'train.txt',
        'test': 'test.txt',
        'modalities': ['T1', 'T2'],

        'paths':{
                # for synth
                'Gen': 'label_maps_generation', 
                'Dmaps': None, 
                'DmapsBag': None, 

                # real images
                'T1': 'T1', 
                'T2': 'T2', 
                'FLAIR': None,
                'CT': None,

                # flip_to_orig (rigid) deformation field
                'flip2orig': 'T1_flip2orig_synthsr_reg',

                # processed ground truths 
                'surface': None, #'surfaces', 
                'distance': None,  
                'segmentation': 'label_maps_segmentation',
                'bias_fields': None,
                'pathology': None,
                'pathology_prob': None,
        }
    },

    'AIBL': { 
        'root': '~/data/aibl_crop',
        'pathology_type': None,
        'train': 'train.txt',
        'test': 'test.txt',
        'modalities': ['T1', 'T2', 'FLAIR'],

        'paths':{
                # for synth
                'Gen': 'label_maps_generation', 
                'Dmaps': None, 
                'DmapsBag': None, 

                # real images
                'T1': 'T1', 
                'T2': 'T2', 
                'FLAIR': 'FLAIR',
                'CT': None,

                # flip_to_orig (rigid) deformation field
                'flip2orig': 'T1_flip2orig_synthsr_reg',

                # processed ground truths 
                'surface': None, #'surfaces', 
                'distance': None,  
                'segmentation': 'label_maps_segmentation',
                'bias_fields': None,
                'pathology': None,
                'pathology_prob': None,
        }
    },

    'OASIS': { 
        'root': '~/data/oasis3_crop',
        'pathology_type': None,
        'train': 'train.txt',
        'test': 'test.txt',
        'modalities': ['T1', 'CT'],

        'paths':{
                # for synth
                'Gen': 'label_maps_generation', 
                'Dmaps': None, 
                'DmapsBag': None, 

                # real images
                'T1': 'T1', 
                'T2': None, 
                'FLAIR': None,
                'CT': 'CT',

                # flip_to_orig (rigid) deformation field
                'flip2orig': 'T1_flip2orig_synthsr_reg',

                # processed ground truths 
                'surface': None, #'surfaces', 
                'distance': None,  
                'segmentation': 'label_maps_segmentation',
                'bias_fields': None,
                'pathology': None,
                'pathology_prob': None,
        }
    },

    'ADNI': { 
        'root': '~/data/adni_crop',
        'pathology_type': None, #'wmh',
        'train': 'train.txt',
        'test': 'test.txt',
        'modalities': ['T1'],

        'paths':{
                # for synth
                'Gen': 'label_maps_generation', 
                'Dmaps': 'Dmaps', 
                'DmapsBag': 'DmapsBag', 

                # real images
                'T1': 'T1', 
                'T2': None, 
                'FLAIR': None,
                'CT': None,

                # flip_to_orig (rigid) deformation field
                'flip2orig': 'T1_flip2orig_synthsr_reg',

                # processed ground truths
                'surface': 'surfaces',  
                'distance': 'Dmaps',  
                'segmentation': 'label_maps_segmentation',
                'bias_fields': None,
                'pathology': 'pathology_maps_segmentation',
                'pathology_prob': 'pathology_probability',
        }
    },

    'ADNI3': { 
        'root': '~/data/adni3_crop',
        'pathology_type': None, # 'wmh',
        'train': 'train.txt',
        'test': 'test.txt',
        'modalities': ['T1', 'FLAIR'],

        'paths':{
                # for synth
                'Gen': 'label_maps_generation', 
                'Dmaps': None, 
                'DmapsBag': None, 

                # real images
                'T1': 'T1', 
                'T2': None, 
                'FLAIR': 'FLAIR',
                'CT': None,

                # flip_to_orig (rigid) deformation field
                'flip2orig': 'T1_flip2orig_synthsr_reg',

                # processed ground truths 
                'surface': None, #'surfaces',  TODO
                'distance': None,  
                'segmentation': 'label_maps_segmentation',
                'bias_fields': None,
                'pathology': 'pathology_maps_segmentation',
                'pathology_prob': 'pathology_probability',
        }
    },

    'ATLAS': { 
        'root': '~/data/atlas_crop',
        'pathology_type': 'stroke',
        'train': 'train.txt',
        'test': 'test.txt',
        'modalities': ['T1'],

        'paths':{
                # for synth
                'Gen': 'label_maps_generation', 
                'Dmaps': None, 
                'DmapsBag': None, 

                # real images
                'T1': 'T1', 
                'T2': None, 
                'FLAIR': None,
                'CT': None,

                # flip_to_orig (rigid) deformation field
                'flip2orig': 'T1_flip2orig_synthsr_reg',

                # processed ground truths 
                'surface': None, #'surfaces',  TODO
                'distance': None,  
                'segmentation': 'label_maps_segmentation',
                'bias_fields': None,
                'pathology': 'pathology_maps_segmentation',
                'pathology_prob': 'pathology_probability',
        }
    },

    'ISLES': { 
        'root': '~/data/isles2022_crop',
        'pathology_type': 'stroke',
        'train': 'train.txt',
        'test': 'test.txt',
        'modalities': ['FLAIR'],

        'paths':{
                # for synth
                'Gen': 'label_maps_generation', 
                'Dmaps': None, 
                'DmapsBag': None, 

                # real images
                'T1': None, 
                'T2': None, 
                'FLAIR': 'FLAIR',
                'CT': None,

                # flip_to_orig (rigid) deformation field
                'flip2orig': 'FLAIR_flip2orig_synthsr_reg',

                # processed ground truths 
                'surface': None, #'surfaces', TODO
                'distance': None,  
                'segmentation': 'label_maps_segmentation',
                'bias_fields': None,
                'pathology': 'pathology_maps_segmentation',
                'pathology_prob': 'pathology_probability',
        }
    },
}


all_dataset_names = dataset_setups.keys()


# get all pathologies
pathology_paths = []
pathology_prob_paths = []
for name, dict in dataset_setups.items():
    # TODO: select what kind of shapes?
    if dict['paths']['pathology'] is not None and dict['pathology_type'] is not None and dict['pathology_type'] == 'stroke':   
        pathology_paths += glob.glob(os.path.join(dict['root'], dict['paths']['pathology'], '*.nii.gz')) \
                                        + glob.glob(os.path.join(dict['root'], dict['paths']['pathology'], '*.nii'))
        pathology_prob_paths += glob.glob(os.path.join(dict['root'], dict['paths']['pathology_prob'], '*.nii.gz')) \
                                        + glob.glob(os.path.join(dict['root'], dict['paths']['pathology_prob'], '*.nii'))
n_pathology = len(pathology_paths)

# with csf
label_list_segmentation = [0,14,15,16,24,77,85,   2, 3, 4, 7, 8, 10,11,12,13,17,18,26,28,   41,42,43,46,47,49,50,51,52,53,54,58,60] # 33
n_neutral_labels = 7







synth_dataset_root = '~/data/synth_anomaly'
synth_dataset_synthpathol_root = '~/data/synth_anomaly_synth'

synth_dataset_setups = { 

    'orig-T1': { 
        'root': os.path.join(synth_dataset_root, 'orig-T1'),
        'pathology_type': 'from_real',
        'test': 'test.txt',
        'modalities': ['synth', 'T1', 'T2', 'FLAIR', 'CT'],

        'paths':{
                'synth': 'synth', 
                # original real (healthy) images
                'T1': 'T1', 
                'T2': 'T2', 
                'FLAIR': 'FLAIR',
                'CT': 'CT',

                # flip_to_orig (rigid) 
                'flip2orig':  'synth_flip2orig',

                # processed ground truths  
                'pathology': 'pathology_maps_segmentation',
                'pathology_prob': 'pathology_probability',
        }
    },

    'orig-T2': { 
        'root': os.path.join(synth_dataset_root, 'orig-T2'),
        'pathology_type': 'from_real',
        'test': 'test.txt',
        'modalities': ['synth', 'T1', 'T2', 'FLAIR', 'CT'],

        'paths':{
                'synth': 'synth', 
                # original real (healthy) images
                'T1': 'T1', 
                'T2': 'T2', 
                'FLAIR': 'FLAIR',
                'CT': 'CT',

                # flip_to_orig (rigid) 
                'flip2orig':  'synth_flip2orig',

                # processed ground truths  
                'pathology': 'pathology_maps_segmentation',
                'pathology_prob': 'pathology_probability',
        }
    },


    'orig-FLAIR': { 
        'root': os.path.join(synth_dataset_root, 'orig-FLAIR'),
        'pathology_type': 'from_real',
        'test': 'test.txt',
        'modalities': ['synth', 'T1', 'T2', 'FLAIR', 'CT'],

        'paths':{
                'synth': 'synth',  
                # original real (healthy) images
                'T1': 'T1', 
                'T2': 'T2', 
                'FLAIR': 'FLAIR',
                'CT': 'CT',

                # flip_to_orig (rigid) 
                'flip2orig':  'synth_flip2orig',

                # processed ground truths  
                'pathology': 'pathology_maps_segmentation',
                'pathology_prob': 'pathology_probability',
        }
    },

    'orig-CT': { 
        'root': os.path.join(synth_dataset_root, 'orig-CT'),
        'pathology_type': 'from_real',
        'test': 'test.txt',
        'modalities': ['synth', 'T1', 'T2', 'FLAIR', 'CT'],

        'paths':{
                'synth': 'synth', 
                'synth_flip2orig': 'synth_flip2orig', 
                # original real (healthy) images
                'T1': 'T1', 
                'T2': 'T2', 
                'FLAIR': 'FLAIR',
                'CT': 'CT',

                # flip_to_orig (rigid) 
                'flip2orig':  'synth_flip2orig',

                # processed ground truths  
                'pathology': 'pathology_maps_segmentation',
                'pathology_prob': 'pathology_probability',
        }
    },

    'orig-Synth': { 
        'root': os.path.join(synth_dataset_root, 'orig-Synth'),
        'pathology_type': 'from_real',
        'test': 'test.txt',
        'modalities': ['synth', 'T1', 'T2', 'FLAIR', 'CT'],

        'paths':{
                'synth': 'synth', 
                'synth_flip': 'synth_flip', 
                # original real (healthy) images
                'T1': 'T1', 
                'T2': 'T2', 
                'FLAIR': 'FLAIR',
                'CT': 'CT',

                # flip_to_orig (rigid) 
                'flip2orig':  'synth_flip2orig',

                # processed ground truths  
                'pathology': 'pathology_maps_segmentation',
                'pathology_prob': 'pathology_probability',
        }
    },


    ###############################
    ###############################


    'orig-T1-synthpathol': { 
        'root': os.path.join(synth_dataset_synthpathol_root, 'orig-T1'),
        'pathology_type': 'from_synth',
        'test': 'test.txt',
        'modalities': ['synth', 'T1', 'T2', 'FLAIR', 'CT'],

        'paths':{
                'synth': 'synth', 
                # original real (healthy) images
                'T1': 'T1', 
                'T2': 'T2', 
                'FLAIR': 'FLAIR',
                'CT': 'CT',

                # flip_to_orig (rigid) 
                'flip2orig':  'synth_flip2orig',

                # processed ground truths  
                'pathology': 'pathology_maps_segmentation',
                'pathology_prob': 'pathology_probability',
        }
    },

    'orig-T2-synthpathol': { 
        'root': os.path.join(synth_dataset_synthpathol_root, 'orig-T2'),
        'pathology_type': 'from_synth',
        'test': 'test.txt',
        'modalities': ['synth', 'T1', 'T2', 'FLAIR', 'CT'],

        'paths':{
                'synth': 'synth', 
                # original real (healthy) images
                'T1': 'T1', 
                'T2': 'T2', 
                'FLAIR': 'FLAIR',
                'CT': 'CT',

                # flip_to_orig (rigid) 
                'flip2orig':  'synth_flip2orig',

                # processed ground truths  
                'pathology': 'pathology_maps_segmentation',
                'pathology_prob': 'pathology_probability',
        }
    },


    'orig-FLAIR-synthpathol': { 
        'root': os.path.join(synth_dataset_synthpathol_root, 'orig-FLAIR'),
        'pathology_type': 'from_synth',
        'test': 'test.txt',
        'modalities': ['synth', 'T1', 'T2', 'FLAIR', 'CT'],

        'paths':{
                'synth': 'synth',  
                # original real (healthy) images
                'T1': 'T1', 
                'T2': 'T2', 
                'FLAIR': 'FLAIR',
                'CT': 'CT',

                # flip_to_orig (rigid) 
                'flip2orig':  'synth_flip2orig',

                # processed ground truths  
                'pathology': 'pathology_maps_segmentation',
                'pathology_prob': 'pathology_probability',
        }
    },

    'orig-CT-synthpathol': { 
        'root': os.path.join(synth_dataset_synthpathol_root, 'orig-CT'),
        'pathology_type': 'from_synth',
        'test': 'test.txt',
        'modalities': ['synth', 'T1', 'T2', 'FLAIR', 'CT'],

        'paths':{
                'synth': 'synth', 
                'synth_flip2orig': 'synth_flip2orig', 
                # original real (healthy) images
                'T1': 'T1', 
                'T2': 'T2', 
                'FLAIR': 'FLAIR',
                'CT': 'CT',

                # flip_to_orig (rigid) 
                'flip2orig':  'synth_flip2orig',

                # processed ground truths  
                'pathology': 'pathology_maps_segmentation',
                'pathology_prob': 'pathology_probability',
        }
    },

    'orig-Synth-synthpathol': { 
        'root': os.path.join(synth_dataset_synthpathol_root, 'orig-Synth'),
        'pathology_type': 'from_synth',
        'test': 'test.txt',
        'modalities': ['synth', 'T1', 'T2', 'FLAIR', 'CT'],

        'paths':{
                'synth': 'synth', 
                'synth_flip': 'synth_flip', 
                # original real (healthy) images
                'T1': 'T1', 
                'T2': 'T2', 
                'FLAIR': 'FLAIR',
                'CT': 'CT',

                # flip_to_orig (rigid) 
                'flip2orig':  'synth_flip2orig',

                # processed ground truths  
                'pathology': 'pathology_maps_segmentation',
                'pathology_prob': 'pathology_probability',
        }
    },

}