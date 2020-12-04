"""This is for training imagenet with random crops
https://gist.github.com/rstml/bbd491287efc24133b90d4f7f3663905
"""


import random
from keras.applications.imagenet_utils import preprocess_input
import keras.preprocessing.image
from PIL import Image as PILimage

def scale(img, resample, min_sz=256, max_sz=480):
    """
    based on: https://github.com/facebook/fb.resnet.torch/blob/master/datasets/transforms.lua#L130
    :param img: PIL object
    :param resample: based on PIL_INTERPOLATION_METHODS
    :param min_sz: lower bound for random sampling
    :param max_sz: lower bound for random sampling
    :return: randomly scaled image along the shorter side
    """
    width = img.size[1]
    height = img.size[0]

    targetSz = random.randint(min_sz, max_sz)
    targetW, targetH = targetSz,targetSz
    if width < height:
        targetH = round(height / width * targetW)
    else:
        targetW = round(width / height * targetH)

    return img.resize((targetH, targetW), resample= resample)



def load_and_crop_img(path, grayscale=False, color_mode='rgb', target_size=None,
                      interpolation='nearest'):
    """Wraps keras_preprocessing.image.utils.loag_img() and adds cropping.
    Cropping method enumarated in interpolation
    # Arguments
        path: Path to image file.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation and crop methods used to resample and crop the image
            if the target size is different from that of the loaded image.
            Methods are delimited by ":" where first part is interpolation and second is crop
            e.g. "lanczos:random".
            Supported interpolation methods are "nearest", "bilinear", "bicubic", "lanczos",
            "box", "hamming" By default, "nearest" is used.
            Supported crop methods are "none", "center", "random".
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """

    # Decode interpolation string. Allowed Crop methods: none, center, random
    interpolation, crop = interpolation.split(":") if ":" in interpolation else (interpolation, "none")

    if crop == "none":
        return keras.preprocessing.image.load_img(path,
                                                        grayscale=grayscale,
                                                        color_mode=color_mode,
                                                        target_size=target_size,
                                                        interpolation=interpolation)

    # Load original size image using Keras
    img = keras.preprocessing.image.load_img(path,
                                                   grayscale=grayscale,
                                                   color_mode=color_mode,
                                                   target_size=None,
                                                   interpolation=interpolation)

    if interpolation not in keras.preprocessing.keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS:
        raise ValueError(
            'Invalid interpolation method {} specified. Supported '
            'methods are {}'.format(interpolation,
                                    ", ".join(
                                        keras.preprocessing.image.utils._PIL_INTERPOLATION_METHODS.keys())))

    resample = keras.preprocessing.keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS[interpolation]

    if crop.startswith('10crop'):
        #dims = [224, 256, 384, 480, 640]
        #loc = ['UL','UR','LL','LR', 'C']
        #flip = ['orig','flip']
        comps = crop.split('-')
        dim, loc, flip = float(comps[1]), comps[2], comps[3]

        img_resized = scale(img, resample, min_sz=dim, max_sz=dim+1)
        if flip == 'flip':
            img_resized = img_resized.transpose(PILimage.FLIP_LEFT_RIGHT)
        w, h = img_resized.width, img_resized.height

        if loc == 'C':
            #center
            img = img_resized.crop((int(round(w / 2)) - int(round(target_size[1] / 2)),
                                    int(round(h / 2)) - int(round(target_size[0] / 2)),
                                    int(round(w / 2)) - int(round(target_size[1] / 2)) + target_size[1],
                                    int(round(h / 2)) - int(round(target_size[0] / 2)) + target_size[0]))
        elif loc == 'UL':
            img = img_resized.crop((0,0, target_size[1], target_size[0]))

        elif loc == 'UR':
            img = img_resized.crop((w - target_size[1], 0, w, target_size[0]))
        elif loc == 'LL':
            img = img_resized.crop((0, h - target_size[0], target_size[1], h))

        elif loc == 'LR':
            img = img_resized.crop((w - target_size[0], h - target_size[0], w, h))
        else:
            raise ValueError('This location {} is not valid', loc)

        return img

    elif crop == 'trainResnet':
        img_resized = scale(img, resample, min_sz=256, max_sz=480)
        w, h = img_resized.width, img_resized.height
        x, y = random.randint(0, w - target_size[1]), random.randint(0,h-target_size[0])
        img = img_resized.crop((x, y, x+target_size[1], y+target_size[0]))

        return img

    elif crop == 'valResnet':
        # This is a mini version of 10crop
        #dims = [224, 256, 384, 480, 640]
        dim = 256
        #locs = ['UL','UR','C','LL','LR']
        loc = 'C'
        #dim = dims[random.randint(0,len(dims)-1)]
        img_resized = scale(img, resample, min_sz=dim, max_sz=dim + 1)

        w, h = img_resized.width, img_resized.height

        if loc == 'C':
            # center
            img = img_resized.crop((int(round(w / 2)) - int(round(target_size[1] / 2)),
                                    int(round(h / 2)) - int(round(target_size[0] / 2)),
                                    int(round(w / 2)) - int(round(target_size[1] / 2)) + target_size[1],
                                    int(round(h / 2)) - int(round(target_size[0] / 2)) + target_size[0]))

        return img

    else:
        # This was used for the first training on Imagenet
        if crop == 'random':
            # Random scaling
            img = scale(img, resample, min_sz=256, max_sz=480)

        # Crop fraction of total image
        crop_fraction = 0.875  # This is the same as Davide used
        target_width = target_size[1]
        target_height = target_size[0]

        if target_size is not None:
            if img.size != (target_width, target_height):

                if crop not in ["center", "random"]:
                    raise ValueError('Invalid crop method {} specified.', crop)

                if interpolation not in keras.preprocessing.keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS:
                    raise ValueError(
                        'Invalid interpolation method {} specified. Supported '
                        'methods are {}'.format(interpolation,
                                                ", ".join(
                                                    keras.preprocessing.keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS.keys())))

                resample = keras.preprocessing.keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS[interpolation]

                width, height = img.size

                # Resize keeping aspect ratio
                # result shold be no smaller than the targer size, include crop fraction overhead
                target_size_before_crop = (target_width / crop_fraction, target_height / crop_fraction)
                ratio = max(target_size_before_crop[0] / width, target_size_before_crop[1] / height)
                target_size_before_crop_keep_ratio = int(width * ratio), int(height * ratio)
                img = img.resize(target_size_before_crop_keep_ratio, resample=resample)

                width, height = img.size

                if crop == "center":
                    left_corner = int(round(width / 2)) - int(round(target_width / 2))
                    top_corner = int(round(height / 2)) - int(round(target_height / 2))
                    img =  img.crop((left_corner, top_corner, left_corner + target_width, top_corner + target_height))
                elif crop == "random":
                    left_shift = random.randint(0, int((width - target_width)))
                    down_shift = random.randint(0, int((height - target_height)))
                    img = img.crop((left_shift, down_shift, target_width + left_shift, target_height + down_shift))

            #subtract the mean pixel per feature map & swap from RGB to BGR
            #img = preprocess_input(keras.preprocessing.image.img_to_array(img), mode='caffe')
        return img


# Monkey patch
#keras.preprocessing.image.DirectoryIterator.load_img = load_and_crop_img