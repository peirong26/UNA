# ported from https://github.com/pvigier/perlin-numpy
# https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/perlin-noise-part-2/perlin-noise-computing-derivatives.html
# https://rtouti.github.io/graphics/perlin-noise-algorithm


import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import torch
import numpy as np 
from FluidAnomaly.misc import stream_3D
    
seed = int(time.time())
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed) 

def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)


def generate_perlin_noise_3d(
        shape, res, tileable=(False, False, False),
        interpolant=interpolant, percentile=None,
):
    """Generate a 3D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of three ints).
            This must be a multiple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of three ints). Note shape must be a multiple
            of res.
        tileable: If the noise should be tileable along each axis
            (tuple of three bools). Defaults to (False, False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """
    seed = int(time.time())
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) 

    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1],0:res[2]:delta[2]]
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1],0:res[2]:delta[2]]
    grid = grid.transpose(1, 2, 3, 0) % 1
    # Gradients
    theta = 2*np.pi*np.random.rand(res[0] + 1, res[1] + 1, res[2] + 1)
    phi = 2*np.pi*np.random.rand(res[0] + 1, res[1] + 1, res[2] + 1)
    gradients = np.stack(
        (np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)),
        axis=3
    )
    if tileable[0]:
        gradients[-1,:,:] = gradients[0,:,:]
    if tileable[1]:
        gradients[:,-1,:] = gradients[:,0,:]
    if tileable[2]:
        gradients[:,:,-1] = gradients[:,:,0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g000 = gradients[    :-d[0],    :-d[1],    :-d[2]]
    g100 = gradients[d[0]:     ,    :-d[1],    :-d[2]]
    g010 = gradients[    :-d[0],d[1]:     ,    :-d[2]]
    g110 = gradients[d[0]:     ,d[1]:     ,    :-d[2]]
    g001 = gradients[    :-d[0],    :-d[1],d[2]:     ]
    g101 = gradients[d[0]:     ,    :-d[1],d[2]:     ]
    g011 = gradients[    :-d[0],d[1]:     ,d[2]:     ]
    g111 = gradients[d[0]:     ,d[1]:     ,d[2]:     ]
    # Ramps
    #print(grid.shape, g000.shape)
    n000 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g000, 3)
    n100 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g100, 3)
    n010 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g010, 3)
    n110 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g110, 3)
    n001 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g001, 3)
    n101 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g101, 3)
    n011 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g011, 3)
    n111 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g111, 3)
    # Interpolation
    t = interpolant(grid)
    n00 = n000*(1-t[:,:,:,0]) + t[:,:,:,0]*n100
    n10 = n010*(1-t[:,:,:,0]) + t[:,:,:,0]*n110
    n01 = n001*(1-t[:,:,:,0]) + t[:,:,:,0]*n101
    n11 = n011*(1-t[:,:,:,0]) + t[:,:,:,0]*n111
    n0 = (1-t[:,:,:,1])*n00 + t[:,:,:,1]*n10
    n1 = (1-t[:,:,:,1])*n01 + t[:,:,:,1]*n11

    noise = ((1-t[:,:,:,2])*n0 + t[:,:,:,2]*n1)
    if percentile is None:
        return noise
    shres = np.percentile(noise, percentile) 
    mask = np.zeros_like(noise) 
    mask[noise >= shres] = 1.
    noise *= mask
    return noise, mask



def generate_fractal_noise_3d(
        shape, res, octaves=1, persistence=0.5, lacunarity=2,
        tileable=(False, False, False), interpolant=interpolant, percentile=None,
):
    """Generate a 3D numpy array of fractal noise.

    Args:
        shape: The shape of the generated array (tuple of three ints).
            This must be a multiple of lacunarity**(octaves-1)*res.
        res: The number of periods of noise to generate along each
            axis (tuple of three ints). Note shape must be a multiple of
            (lacunarity**(octaves-1)*res).
        octaves: The number of octaves in the noise. Defaults to 1.
        persistence: The scaling factor between two octaves.
        lacunarity: The frequency factor between two octaves.
        tileable: If the noise should be tileable along each axis
            (tuple of three bools). Defaults to (False, False, False).
        interpolant: The, interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of fractal noise and of shape generated by
        combining several octaves of perlin noise.

    Raises:
        ValueError: If shape is not a multiple of
            (lacunarity**(octaves-1)*res).
    """
    seed = int(time.time())
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) 
    
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_3d(
            shape,
            (frequency*res[0], frequency*res[1], frequency*res[2]),
            tileable,
            interpolant
        )
        frequency *= lacunarity
        amplitude *= persistence

    if percentile is None:
        return noise
    shres = np.percentile(noise, percentile) 
    mask = np.zeros_like(noise) 
    mask[noise >= shres] = 1.
    noise *= mask
    return noise, mask


def generate_shape_3d(shape, perlin_res, percentile, device):
    pprob, p = generate_perlin_noise_3d(shape, perlin_res, tileable=(True, False, False), percentile=percentile) 
    return torch.from_numpy(p).to(device), torch.from_numpy(pprob).to(device)


def generate_velocity_3d(shape, perlin_res, V_multiplier, device, save_orig_for_visualize = False): 
    # shape must be a multiple of 2
    #pad_shape = [ (x + 1) // 2 * 2 for x in shape ]
    pad_shape = [ 200, 200, 200 ]
    #print('pad', shape, '->', pad_shape)

    # generate random potentials (back to original shape)
    curl_a = generate_perlin_noise_3d(pad_shape, perlin_res, tileable=(True, False, False)) 
    curl_b = generate_perlin_noise_3d(pad_shape, perlin_res, tileable=(True, False, False)) 
    curl_c = generate_perlin_noise_3d(pad_shape, perlin_res, tileable=(True, False, False)) 

    # back to original shape
    curl_a = curl_a[(pad_shape[0] - shape[0]) // 2 : (pad_shape[0] - shape[0]) // 2 + shape[0], \
                    (pad_shape[1] - shape[1]) // 2 : (pad_shape[1] - shape[1]) // 2 + shape[1], \
                    (pad_shape[2] - shape[2]) // 2 : (pad_shape[2] - shape[2]) // 2 + shape[2]
                    ]
    curl_b = curl_b[(pad_shape[0] - shape[0]) // 2 : (pad_shape[0] - shape[0]) // 2 + shape[0], \
                    (pad_shape[1] - shape[1]) // 2 : (pad_shape[1] - shape[1]) // 2 + shape[1], \
                    (pad_shape[2] - shape[2]) // 2 : (pad_shape[2] - shape[2]) // 2 + shape[2]
                    ]
    curl_c = curl_c[(pad_shape[0] - shape[0]) // 2 : (pad_shape[0] - shape[0]) // 2 + shape[0], \
                    (pad_shape[1] - shape[1]) // 2 : (pad_shape[1] - shape[1]) // 2 + shape[1], \
                    (pad_shape[2] - shape[2]) // 2 : (pad_shape[2] - shape[2]) // 2 + shape[2]
                    ]

    Vx, Vy, Vz = stream_3D(torch.from_numpy(curl_a).to(device), 
                            torch.from_numpy(curl_b).to(device), 
                            torch.from_numpy(curl_c).to(device))
    
    if save_orig_for_visualize:
        return {'Vx': (Vx * V_multiplier), 'Vy': (Vy * V_multiplier).to(device), 'Vz': (Vz * V_multiplier), 
                'curl_a': curl_a, 'curl_b': curl_b, 'curl_c': curl_c}
    return {'Vx': (Vx * V_multiplier), 'Vy': (Vy * V_multiplier).to(device), 'Vz': (Vz * V_multiplier)}


