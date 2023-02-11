import numpy as np
from numpy import ndarray
from skimage.segmentation import slic, felzenszwalb, watershed


ACCEPTED_SEG_TYPE = ['slic', 'felzen', 'watershed', 'bass']


def get_seg_map(seg_type: str, image: ndarray, *args, **kwargs) -> ndarray:
    """
    Called helper function. This functions provides documentation on all other 
        useful segmentation functions.

    Use the following psuedo code to understand how to use the various segmentation functions.

    if seg_type == 'slic':
        seg_map = slic(image, *args, **kwargs)

        For args and kwargs for 'slic', See documentation on:
        https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.slic

    elif seg_type == 'felzen':
        seg_map = felzenszwalb(image, *args, **kwargs)

        For args and kwargs for 'felzen', See documentation on:
        https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb

    elif seg_type == 'watershed':
        seg_map = watershed(image, *args, **kwargs)

        For args and kwargs for 'watershed', See documentation on:
        https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed

    elif seg_type == 'bass':
        Must have a build from https://github.com/BGU-CS-VIL/BASS

        If used, this function will print out information on how to perform bass segmentation with the
        cse.bass_segmentation module.

        returns None

    return seg_map


    Args:
        seg_type (str): type of segmentations to use. Must be 'slic', 'felzen',
            'watershed', or 'bass'

        image (ndarray): Image to be segmented

        *args: positional arguments to be passed along to appropriate seg function.

        **kwargs: keyword arguments to be passed along to appropriate seg function.

    Returns:
        ndarray: segmentation map
    """

    assertion_str = f'{seg_type} is not accepted.\nPlease use one of the following {ACCEPTED_SEG_TYPE}'
    assert seg_type in ACCEPTED_SEG_TYPE, assertion_str

    seg_map = None

    if seg_type == 'slic':
        seg_map = slic(image, *args, **kwargs)

    elif seg_type == 'felzen':
        seg_map = felzenszwalb(image, *args, **kwargs)

    elif seg_type == 'watershed':
        seg_map = watershed(image, *args, **kwargs)

    elif seg_type == 'bass':
        print('To use bass segmentation, see documentation on cse.bass_segmentation.')
        print('see cse.bass_segmentation.__doc__ for details')
        return

    assertion_str = f'Type {type(seg_map)} is not a numpy array'
    assert isinstance(seg_map, ndarray), assertion_str

    return seg_map
