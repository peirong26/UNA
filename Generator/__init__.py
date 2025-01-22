
"""
Datasets interface.
"""
from .constants import dataset_setups
from .datasets import BaseGen, UNAGen



dataset_options = {  
    'default': BaseGen,
    'brain_id': UNAGen,
}




def build_datasets(gen_args, device):
    """Helper function to build dataset for different splits ('train' or 'test')."""
    datasets = {}
    for dataset_name in gen_args.dataset_names:
        dataset_path_dict = dataset_setups[dataset_name]
        datasets[dataset_name] = dataset_options[gen_args.dataset_option](gen_args, dataset_name, dataset_path_dict, device)
    return datasets

