"""
Backbone modules.
"""

from Trainer.models.unet3d.model import UNet3D, UNet2D, UNet3DSep


backbone_options = {
    'unet2d': UNet2D,
    'unet3d': UNet3D,
    'unet3d_2stage': UNet3D,
    'unet3d_sep': UNet3DSep,
}



####################################


def build_backbone(args, backbone, num_cond=0):
    backbone = backbone_options[backbone](args.in_channels + num_cond, args.f_maps, 
                                            args.layer_order, args.num_groups, args.num_levels, 
                                            args.unit_feat,
                                            )
    return backbone
 