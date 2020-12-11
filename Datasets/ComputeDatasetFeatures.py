from pycocotools.coco import COCO
from pycocotools.mask import area,encode
from asn.utils import load_pickle, save_pickle
import numpy as np
from copy import deepcopy

from scipy.ndimage import zoom
from scipy.misc import logsumexp
from scipy.io import loadmat
import skimage.io as io
import tensorflow as tf
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import os


def obtainSaliencyMaps(imgs, model='DeepGazeII',deepGazeDir ='/mnt/Googolplex/PycharmProjects/ASN_mac/COCO/deep_gaze/'):
    """
    This is adapted from https://deepgaze.bethgelab.org/
    :param imgs: BHWC, three channels (RGB)
    :param model:  DeepGaze II or ICF
    :param deepGazeDir: where the checkpoints are.
    :return: log_density_prediction
    """
    # based on Demo of deep_gaze
    # load precomputed log density over a 1024x1024 image
    centerbias_template = np.load(deepGazeDir + 'centerbias.npy')
    # rescale to match image size
    centerbias = zoom(centerbias_template, (imgs.shape[1] / 1024, imgs.shape[2] / 1024), order=0, mode='nearest')
    # renormalize log density
    centerbias -= logsumexp(centerbias)
    centerbias_data = centerbias[np.newaxis, :, :, np.newaxis]

    #%%
    tf.reset_default_graph()

    check_point = deepGazeDir + model + '.ckpt'  # DeepGaze II or ICF

    new_saver = tf.train.import_meta_graph('{}.meta'.format(check_point))

    input_tensor = tf.get_collection('input_tensor')[0]
    centerbias_tensor = tf.get_collection('centerbias_tensor')[0]
    log_density = tf.get_collection('log_density')[0]

    with tf.Session() as sess:
        new_saver.restore(sess, check_point)

        log_density_prediction = sess.run(log_density, {
            input_tensor: imgs,
            centerbias_tensor: centerbias_data,
        })

    return log_density_prediction

def obtainSceneComplexity(imgIds, dataset_name = '1_street', dataDir = '/home/lynn/PycharmProjects/ASN_mac/COCO/Dataset/'):
    """
    :param imgIds: Id of the images
    :param dataset: name of the dataset
    :param dataDir: loaction of the matlab output from the CESC package by Iris
    :return: Ordered contrast energy & spatial coherence values (gray, blue-yellow, red-green)
    """

    sceneComplexity = loadmat(dataDir+ dataset_name + '/SceneComplexity/WB_LGNlarge.mat')
    CE = np.zeros((len(imgIds),3))
    SC = np.zeros((len(imgIds),3))

    for i in range(len(imgIds)):
        id = imgIds[i]
        idx = [id == filename for filename in sceneComplexity['filenames']]
        CE[i, :] = sceneComplexity['CE'][idx,:]
        SC[i, :] = sceneComplexity['SC'][idx,:]

    return CE, SC

def obtainImageFeatures(dataset_name, path, annFile, annFileStuff, show=False, compute_saliency = True):
    dataset = load_pickle(dataset_name)

    coco = COCO(annFile)
    cocoStuff = COCO(annFileStuff)

    imgs = coco.loadImgs(dataset['img_ids'])

    sceneComplexity = loadmat(path + '/SceneComplexity/WB_LGNlarge.mat')

    stuff_cats = cocoStuff.cats
    super_cats_nms = sorted(set([stuff_cats[92 + c]['supercategory'] for c in range(len(stuff_cats))]))

    if os.path.exists(path + 'Features_single_radial.pickle'):
        features = load_pickle(path + 'Features_single_radial.pickle')
        temp = load_pickle(path + 'Temp_features.pickle')

        target_centres_x_BBox = temp['target_centres_x_BBox']
        target_centres_y_BBox = temp['target_centres_y_BBox']

        target_centres_x_CoM =  temp['target_centres_x_CoM']
        target_centres_y_CoM = temp['target_centres_y_CoM']
        target_category = temp['target_category']

        stuff_collection = temp['stuff_collection']
        super_stuff_collection = temp['super_stuff_collection']

    else:
    #targetComplexity = loadmat(path + 'TargetComplexity/WB_LGNlarge.mat')

        features = deepcopy(dataset)
        features['dataset_name'] = dataset_name  # D
        features['targetCategory'] = {}  # D
        features['targetCategoryId'] = {}  # D
        features['salienceMap'] = {}  # D
        features['targetMask'] = {}
        features['targetSaliencePeak'] = {}  # D
        features['targetSalienceSum'] = {}  # D
        features['targetSalienceAvg'] = {}
        features['targetArea'] = {}  # D
        features['targetCentreBBox'] = {}  # D
        features['targetCentreoM'] = {}  # D
        #features['targetEccentricityHorizontal'] = {}  # D
        features['targetEccentricityCentreBBox'] = {}  # D
        features['targetEccentricityCentreoM'] = {}
        features['targetLocationTypicalityDatatset_BBox'] = {}  # D
        features['targetLocationTypicalityCategory_BBox'] = {}  # D
        features['targetLocationTypicalityDatatset_CoM'] = {}
        features['targetLocationTypicalityCategory_CoM'] = {}
        features['sceneComplexity_CE'] = {}  # D
        features['sceneComplexity_SC'] = {}  # D
        features['sceneStuff'] = {}  # D
        features['sceneStuff_super'] = {}  # D
        features['sceneStuffTypicalityDataset'] = {}
        features['sceneStuff_superTypicalityDataset'] = {}
        features['sceneStuffTypicalityCategory'] = {}
        features['sceneStuff_superTypicalityCategory'] = {}

        # Collect some stuff over all targets
        target_centres_x_BBox = []
        target_centres_y_BBox = []

        target_centres_x_CoM = []
        target_centres_y_CoM = []
        target_category = []

        stuff_collection = np.zeros((len(imgs) + 1, 92))
        super_stuff_collection = np.zeros(((len(imgs) + 1), len(super_cats_nms)))
        temp = {}

    #%% SALIENCY preparation
    # based on Demo of deep_gaze
    if compute_saliency == True:
        model = 'DeepGazeII'
        deepGazeDir = '/mnt/Googolplex/PycharmProjects/ASN_mac/COCO/deep_gaze/'
        # load precomputed log density over a 1024x1024 image
        centerbias_template = np.load(deepGazeDir + 'centerbias.npy')

        tf.reset_default_graph()
        check_point = deepGazeDir + model + '.ckpt'  # DeepGaze II or ICF

        new_saver = tf.train.import_meta_graph('{}.meta'.format(check_point))

        input_tensor = tf.get_collection('input_tensor')[0]
        centerbias_tensor = tf.get_collection('centerbias_tensor')[0]
        log_density = tf.get_collection('log_density')[0]

    else:
        features['salienceMap'] = load_pickle(path +'SalienceMaps.pickle')


    for i in range(len(features['targetCategory']) - 1, len(imgs)):
        print(str(i) + '/' + str(len(imgs)))
        img = imgs[i]
        stuff_annIds = cocoStuff.getAnnIds(imgIds=img['id'])
        stuff_anns = cocoStuff.loadAnns(stuff_annIds)
        for stuff in stuff_anns:
            if stuff['category_id'] != 183: # get rid of the 'other' cat
                stuff_area = stuff['area']/(img['width'] * img['height'])
                stuff_collection[i,92 - stuff['category_id']] = stuff_area
                # Find corresponding super cat
                super_cat = stuff_cats[stuff['category_id']]['supercategory']
                idx = [super_cat== s for s in super_cats_nms]
                super_stuff_collection[i, np.where(idx)[0][0]] = super_stuff_collection[i, np.where(idx)[0][0]] + stuff_area

        features['sceneStuff'][i] = stuff_collection[i,:]
        features['sceneStuff_super'][i] = super_stuff_collection[i,:]

        ann = coco.loadAnns([dataset['annIds'][i]])[0]

        # Scene complexity
        # Read in scene complexities
        idx = [str(ann['id']) + '_' + img['file_name'] == filename[0][0] for filename in sceneComplexity['filenames']]
        features['sceneComplexity_CE'][i] = sceneComplexity['CE'][np.where(idx)[0][0], :]
        features['sceneComplexity_SC'][i] = sceneComplexity['SC'][np.where(idx)[0][0], :]

        #features['sceneComplexity_CE'][i] = CE[i]
        #features['sceneComplexity_SC'][i] = SC[i]

        # load in the images:
        I = io.imread(img['coco_url'])
        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]
            blackwhite = True
        else:
            blackwhite = False
        d = dataset['d_sel'][i]
        # select the chosen image square
        selection = np.zeros(I.shape, dtype=bool)
        if img['width'] > img['height']:
            selection[:,d:img['width'] - ((img['width'] - img['height']) - d),:] = True
            dim_sel = img['height']
            middle = [d + dim_sel / 2, dim_sel / 2]
        elif imgs[i]['width'] < imgs[i]['height']:
            selection[d:img['height'] - ((img['height'] - img['width']) - d), :,:] = True
            dim_sel = img['width']
            middle = [dim_sel / 2,d + dim_sel / 2]

        else:
            selection[:] = True
            dim_sel = img['width']
            middle = [dim_sel / 2,dim_sel / 2]


        I_sel = I[selection].reshape(1,dim_sel, dim_sel, I.shape[-1])
        #%% SALIENCY MAP
        if compute_saliency == True:
            # rescale to match image size
            centerbias = zoom(centerbias_template, (I_sel.shape[1] / 1024, I_sel.shape[2] / 1024), order=0, mode='nearest')
            # renormalize log density
            centerbias -= logsumexp(centerbias)
            centerbias_data = centerbias[np.newaxis, :, :, np.newaxis]
            if blackwhite == False:
                with tf.Session() as sess:
                    new_saver.restore(sess, check_point)
                    log_density_prediction = sess.run(log_density, {
                        input_tensor: I_sel,
                        centerbias_tensor: centerbias_data,
                    })
                features['salienceMap'][i] = np.exp(log_density_prediction)

        if show == True:
            for j in range(len(features['targetCategory'])):
                i = np.random.randint(len(features['targetCategory']))
                I = io.imread(path + 'val2017_single/' + str(features['x_ids'][i]))
                plt.figure()
                plt.gca().imshow(I, alpha=0.8)
                m = plt.gca().matshow((features['salienceMap'][i][0, :, :, 0]), alpha=0.5, cmap=plt.cm.RdBu)
                plt.colorbar(m)
                targetcats = features['categories'][np.where(features['targetCategory'][i] == features['cat_coco_ids'])[0][0]]
                plt.title(str(targetcats) + ':  ' + str(features['targetSalienceSum'][i]) + ' SC: ' + str(np.round(features['sceneComplexity_SC'][i][0],2)))
                plt.tight_layout()
                plt.savefig(path + 'TargetSalienceSceneComplexity/Example_' + str(i))
                plt.close()
        #%% ANNOTATIONS


        if show == True:
            plt.figure()
            plt.imshow(I[:,:,:])
            plt.axvline(x=middle[0], alpha=0.5, color='red', label='_nolegend_')

            coco.showAnns([ann])
            title = ann['category_id']
            plt.title(str(title))
            plt.show()

        features['targetMask'][i] = {}
        #features['targetCategory'][i] = np.zeros(len(anns))
        #features['targetArea'][i] = np.zeros(len(anns))
        #features['targetEccentricityHorizontal'][i] = np.zeros(len(anns))
        #features['targetEccentricityCentre'][i] = np.zeros(len(anns))
        #features['targetSaliencePeak'][i] = np.zeros(len(anns))
        #features['targetSalienceSum'][i] = np.zeros(len(anns))
        #features['targetSalienceAvg'][i] = np.zeros(len(anns))
        #features['targetCentre'][i] = np.zeros((len(anns),2))
        #features['targetComplexity_CE'][i] = np.zeros((len(anns),3))
        #features['targetComplexity_SC'][i] = np.zeros((len(anns), 3))


        # Record category
        features['targetCategory'][i] = dataset['categories'][np.where(dataset['y'][i]==1)[0][0]] #ann['category_id']

        features['targetCategoryId'][i] = dataset['cat_coco_ids'][
            np.where(dataset['y'][i] == 1)[0][0]]  # ann['category_id']
        target_category.append(features['targetCategory'][i])
        # Compute area
        mask = coco.annToMask(ann)
        mask_sel = mask[selection[:,:,0]].reshape(dim_sel, dim_sel)
        target_area = area(encode(np.asfortranarray(mask_sel)))
        features['targetArea'][i] = target_area/(dim_sel**2)
        features['targetMask'][i] = np.array(mask_sel, dtype=bool)
        if blackwhite == False:
            # Compute mean target saliency
            features['targetSaliencePeak'][i] = np.max(features['salienceMap'][i][:,np.array(mask_sel, dtype=bool),:])
            features['targetSalienceSum'][i] = np.sum(features['salienceMap'][i][:,np.array(mask_sel, dtype=bool),:])
            features['targetSalienceAvg'][i]= features['targetSalienceSum'][i]/target_area



        # Compute eccentricity (bbox: [x,y,width,height])
        idx = np.where(selection == True)
        if ann['bbox'][0] >= np.min(idx[1]):
            start_horz = ann['bbox'][0]
        else:
            start_horz = np.min(idx[1])
        if ann['bbox'][0] + ann['bbox'][2] <= np.max(idx[1]):
            end_horz = (ann['bbox'][0] + ann['bbox'][2])
        else:
            end_horz = np.max(idx[1])
        #features['targetEccentricityHorizontal'][i] = abs(middle[0] - (start_horz + end_horz)/2)/(dim_sel/2) # normalized by image size

        if ann['bbox'][1] >= np.min(idx[0]):
            start_vert = ann['bbox'][1]
        else:
            start_vert = np.min(idx[0])
        if ann['bbox'][1] + ann['bbox'][3] <= np.max(idx[0]):
            end_vert = (ann['bbox'][1] + ann['bbox'][3])
        else:
            end_vert = np.max(idx[0])

        # Compute target center (x, y) according to the Bbox
        features['targetCentreBBox'][i] = np.array([(start_horz + end_horz) / 2 - np.min(idx[1]),
                                                    (start_vert + end_vert) / 2 - np.min(idx[0])]) / dim_sel
        target_centres_x_BBox.append(features['targetCentreBBox'][i][0])
        target_centres_y_BBox.append(features['targetCentreBBox'][i][1])

        features['targetEccentricityCentreBBox'][i] = euclidean([0.5,0.5],features['targetCentreBBox'][i])

        # Compute target center according to the centre of mass
        # This is based on a script by Noor:
        m = mask_sel/np.sum(mask_sel[:])

        # 1. obtain marginal distributions, 2. compute expected values
        features['targetCentreoM'][i] = np.array([np.sum(np.sum(m,axis=0) * np.arange(m.shape[0])),
                                                  np.sum(np.sum(m,axis=1) * np.arange(m.shape[1]))]) /dim_sel

        target_centres_x_CoM.append(features['targetCentreoM'][i][0])
        target_centres_y_CoM.append(features['targetCentreoM'][i][1])
        features['targetEccentricityCentreoM'][i] = euclidean([0.5,0.5],features['targetCentreoM'][i])


        if (np.divmod(i,100)[1] == 0) | (i == len(imgs)-1):
            temp['target_category'] = target_category
            temp['stuff_collection'] = stuff_collection
            temp['super_stuff_collection'] = super_stuff_collection
            temp['target_centres_x_BBox'] = target_centres_x_BBox
            temp['target_centres_y_BBox'] = target_centres_y_BBox
            temp['target_centres_x_CoM'] = target_centres_x_CoM
            temp['target_centres_y_CoM'] = target_centres_y_CoM


            save_pickle(temp, path + 'Temp_features')
            save_pickle(features, path + 'Features_single_radial')

    if compute_saliency == True:
        save_pickle(features['salienceMap'], path + 'SalienceMaps')

    # Compute targetLocationTypicalityDataset: euclidean distance to average
    features['targetCentreBBox']['all'] = np.array([np.mean(np.array(target_centres_x_BBox)), np.mean(np.array(target_centres_y_BBox))])
    features['targetCentreBBox']['categories'] = np.zeros((len(features['cat_coco_ids']),2))

    features['targetCentreoM']['all'] = np.array(
        [np.mean(np.array(target_centres_x_CoM)), np.mean(np.array(target_centres_y_CoM))])
    features['targetCentreoM']['categories'] = np.zeros((len(features['cat_coco_ids']), 2))

    features['sceneStuff']['all'] = np.mean(stuff_collection, axis = 0)
    features['sceneStuff']['all'] = features['sceneStuff']['all'][np.newaxis,:]
    features['sceneStuff']['categories'] = np.zeros((len(features['cat_coco_ids']), stuff_collection.shape[1]))

    features['sceneStuff_super']['all'] =np.mean(super_stuff_collection, axis = 0)
    features['sceneStuff_super']['all'] = features['sceneStuff_super']['all'][np.newaxis,:]
    features['sceneStuff_super']['categories'] = np.zeros((len(features['cat_coco_ids']), super_stuff_collection.shape[1]))

    for c in range(len(features['categories'])):
        cat = features['categories'][c]
        if len(np.where(np.array(target_category)==cat)[0]) >0 :
            current_cat = np.where(np.array(features['categories']) == cat)[0][0]
            cases_targets = np.where(np.array(target_category)==cat)[0]
            features['targetCentreBBox']['categories'][current_cat] = np.array([np.mean(np.array(target_centres_x_BBox)[cases_targets]), np.mean(np.array(target_centres_y_BBox)[cases_targets])])
            features['targetCentreoM']['categories'][current_cat] = np.array(
                [np.mean(np.array(target_centres_x_CoM)[cases_targets]),
                 np.mean(np.array(target_centres_y_CoM)[cases_targets])])

            cases_images = np.where([y[current_cat] == True for y in features['y']])
            features['sceneStuff']['categories'][current_cat] = np.mean(stuff_collection[cases_images,:],axis=1)
            features['sceneStuff_super']['categories'][current_cat] = np.mean(super_stuff_collection[cases_images, :], axis=1)

    for i in range(len(features['x_ids'])):
        features['targetLocationTypicalityDatatset_BBox'][i] = euclidean(features['targetCentreBBox']['all'],
                                                                         features['targetCentreBBox'][i])

        features['targetLocationTypicalityDatatset_CoM'][i] = euclidean(features['targetCentreoM']['all'],
                                                                         features['targetCentreoM'][i])

        current_cat= np.where(np.array(features['categories']) == features['targetCategory'][i])[0][0]
        features['targetLocationTypicalityCategory_BBox'][i] = euclidean(features['targetCentreBBox']['categories']
                                                                         [current_cat,:],
                                                                           features['targetCentreBBox'][i])

        features['targetLocationTypicalityCategory_CoM'][i] = euclidean(
            features['targetCentreoM']['categories'][current_cat, :],
            features['targetCentreoM'][i])

        features['sceneStuffTypicalityDataset'][i] =euclidean(features['sceneStuff']['all'],features['sceneStuff'][i])
        features['sceneStuff_superTypicalityDataset'][i] = euclidean(features['sceneStuff_super']['all'], features['sceneStuff_super'][i])

        features['sceneStuffTypicalityCategory'][i] = euclidean(features['sceneStuff']['categories'][current_cat,:],
                                                                        features['sceneStuff'][i])
        features['sceneStuff_superTypicalityCategory'][i] = euclidean(
                features['sceneStuff_super']['categories'][current_cat, :],
                features['sceneStuff_super'][i])

    save_pickle(features,path + 'Features_single_radial')

    return features

crashed = False
cocoDir = '/mnt/Googolplex/coco/'
stem_dir = '/mnt/Googolplex//PycharmProjects/ASN_mac/COCO/Dataset/'
dataType = 'val2017'
min_area = 0.005
targetType = 'single'
spatial_constraint = 'radial'

annFile = '{}/annotations/instances_redistributed_{}.json'.format(cocoDir, dataType)
annFileStuff = '{}/annotations/stuff_redistributed_{}.json'.format(cocoDir, dataType)

for d in ['1_street']:#['2_food']:#,'2_food']: #'1_street',
    tf.reset_default_graph()
    print(d)
    path = stem_dir + d + '/'
    dataset_name = path + dataType + '_' + targetType + '_'+ spatial_constraint + '.pickle'

    features = obtainImageFeatures(dataset_name, path, annFile, annFileStuff, compute_saliency=True)

"""
if crashed == True:
    features = load_pickle(path + 'Features_single_radial.pickle')
    temp = load_pickle(path + 'Temp_features.pickle')
    for t in temp:
        globals()[t] = temp[t]

    for i in range(len(features['targetCategory'])):
        for h in range(len(dataset['cat_coco_ids'])):  # This is quite ugly but a way to deal with the merged categories.
            curr = False
            if isinstance(dataset['cat_coco_ids'][h], list):
                if isinstance(features['targetCategoryId'][i], list):
                    if (features['targetCategoryId'][i] == dataset['cat_coco_ids'][h]):
                        curr = True

            elif isinstance(features['targetCategoryId'][i], list):
                if isinstance(dataset['cat_coco_ids'][h], list):
                    if (features['targetCategoryId'][i] == dataset['cat_coco_ids'][h]):
                        curr = True
            else:
                if dataset['cat_coco_ids'][h] == features['targetCategoryId'][i]:
                    curr = True
            if curr == True:
                features['targetCategory'][i] = features['categories'][h]
                #targets.append(str(ann['category_id']) + ': ' + dataset['categories'][h])


"""