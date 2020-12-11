import matplotlib
matplotlib.use('Agg')
#matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import os
from pycocotools.coco import COCO
from pycocotools.mask import area,encode
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
import joblib
from copy import deepcopy

pylab.rcParams['figure.figsize'] = (10.0, 8.0)


def makeDataset(dataset_name,dataDir, resultDir, dataType, object_cats ,context_cats=None, stuff_cats=None, merged_cats= None,
                targetType = 'multi', plotting=None, squaring=True, saving=True, make_dataset=True, min_area=0.005,
                spatial_constraint='radial', center_bias = False, computeCoM=False):

    """
    :param dataset_name: Name of the resulting data set
    :param object_cats: Name of target objects
    :param stuff_cats: Name of context annotations for the inital image selection
    :param merged_cats: Join multipe coco cats to a new target category
    :param targetType: Single or multi target task
    :param dataDir: coco directory
    :param dataType: name of dataset to work from e.g. val2017
    :param started: if already started, the existing files will be loaded in
    :param plotting: int, random selection of dataset
    :param squaring: if the target format should be squared or original format
    :param saving: if the outputs should be saved
    :param make_dataset: if images themselves should be saved as a new dataset in the coco directory/images
    :param min_area: the minimum proportion of space an object has to occupy to be an eligible target.

    """

    annFile = '{}/annotations/instances_redistributed_{}.json'.format(dataDir, dataType)
    coco = COCO(annFile)
    cat_Ids = coco.getCatIds(catNms=object_cats)
    cats = coco.loadCats(cat_Ids)
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    ids = [cat['id'] for cat in cats]
    ids = np.array(ids)

    if (stuff_cats != None) & (context_cats != None):
        raise ValueError('It is currently not possible to both limit your image selection to stuff annotations and '
                         'context object categories. Please choose only one.')

    all_img_Ids = []
    if stuff_cats is not None:
        annFileStuff = '{}/annotations/stuff_redistributed_{}.json'.format(dataDir, dataType)
        cocoStuff = COCO(annFileStuff)
        cat_Ids_context = cocoStuff.getCatIds(catNms=stuff_cats)

        cats_stuff = cocoStuff.loadCats(cat_Ids_context)
        nms_stuff = [cat['name'] for cat in cats_stuff]
        print('COCO categories: \n{}\n'.format(' '.join(nms_stuff)))

        ids_stuff = [cat['id'] for cat in cats_stuff]
        ids_stuff = np.array(ids_stuff)

        for cat in cat_Ids_context:  # Find images based on context categories
            all_img_Ids.extend(cocoStuff.getImgIds(catIds=cat))

    elif context_cats is not None:
        cat_Ids_context = coco.getCatIds(catNms=context_cats)
        cats_context = coco.loadCats(cat_Ids_context)
        nms_context = [cat['name'] for cat in cats_context]

        cat_Ids = cat_Ids_context + cat_Ids
        cats = coco.loadCats(cat_Ids)
        nms = [cat['name'] for cat in cats]
        print('COCO categories: \n{}\n'.format(' '.join(nms)))

        ids = [cat['id'] for cat in cats]
        ids = np.array(ids)

        for cat in cat_Ids_context:  # Find images based on context categories
            all_img_Ids.extend(coco.getImgIds(catIds=cat))

    else:
        for cat in cat_Ids:  # Find images based on context categories
            all_img_Ids.extend(coco.getImgIds(catIds=cat))

    # get rid of duplicates:
    all_img_Ids = list(set(all_img_Ids))

    if make_dataset == True:
        if not os.path.isdir(dataDir + 'images/' + dataset_name + '/' + dataType + '_' + targetType + '_' + spatial_constraint + '/'):
            os.mkdir(dataDir + 'images/' + dataset_name + '/' + dataType + '_' + targetType + '_' + spatial_constraint + '/')

        dataset = {}
        dataset['name'] = dataset_name
        dataset['dataType'] = dataType
        dataset['targetType'] = targetType
        if merged_cats:
            dataset['categories'] = []
            dataset['cat_coco_ids'] = []
            redundant = []
            redundant_ids = []
            new = []
            for m in merged_cats:
                redundant.append(merged_cats[m])
                new.append(m)
                for n in range(len(nms)):
                    if nms[n] in merged_cats[m]:
                        redundant_ids.append(ids[n])
            cat_idx = [n not in redundant[0] for n in nms]
            dataset['categories'] = np.array(nms)[np.where(cat_idx)[0]]
            #dataset['categories'].extend(list(set(nms).symmetric_difference(redundant[0])))
            dataset['categories'] = np.append(dataset['categories'],new)

            dataset['cat_coco_ids'].extend(list(set(ids).symmetric_difference(redundant_ids)))
            dataset['cat_coco_ids'].append(redundant_ids)

        else:
            dataset['categories'] = nms
            dataset['cat_coco_ids'] = ids

        dataset['min_area'] = min_area
        dataset['spatial_constraint'] = spatial_constraint
        dataset['center_bias'] = center_bias
        if stuff_cats is not None:
            dataset['stuff'] = nms_stuff
            dataset['stuff_coco_ids'] = ids_stuff
        elif context_cats is not None:
            dataset['context'] = nms_context
            dataset['contexc_coco_ids'] = cat_Ids_context

    d_sel = []

    if str(plotting).isdigit() == True:
        counting = np.random.randint(len(all_img_Ids), size=plotting)
    else:
        counting = range(len(all_img_Ids))

    if os.path.exists(
            resultDir + dataset_name + '/' + str(dataType) + '_' + str(
                    targetType) + '_' + spatial_constraint + '.pickle'):
        dataset = joblib.load(
            resultDir + dataset_name + '/' + str(dataType) + '_' + str(
                targetType) + '_' + spatial_constraint + '.pickle')

        d_sel = dataset['d_sel']

        counting = range(np.where(all_img_Ids == dataset['img_ids'][-1])[0][0]+ 1, len(all_img_Ids))

    for Id in counting:  # np.random.randint(len(all_img_Ids),size=50):#range(len(all_img_Ids)):
        img = coco.loadImgs(all_img_Ids[Id])[0]
        #img = coco.loadImgs(319607)[0]
        print(str(Id) + '/' + str(len(all_img_Ids)) + '. Processing image ' + str(img['id']))

        if squaring == True:
            if img['width'] > img['height']:
                difference = (img['width'] - img['height'])
                min_area_img = min_area * (img['height'] * img['height'])
            elif img['width'] < img['height']:
                difference = (img['height'] - img['width'])
                min_area_img = min_area * (img['width'] * img['width'])
            else:
                difference = 0
                min_area_img = min_area * (img['width'] * img['width'])
        else:
            difference = 0
        anns_plot = []
        if targetType == 'multi':
            annIds = []

            for cat in cat_Ids:
                annIds.extend(
                    coco.getAnnIds(catIds=cat, imgIds=img['id'], areaRng=[min_area_img, np.inf], iscrowd=None))

            # annIds = coco.getAnnIds(imgIds=img['id'], areaRng=[min_area, np.inf],iscrowd=None)
            anns = coco.loadAnns(annIds)

            print('Found ' + str(len(anns)) + ' annotations')

            anns_all = [ann['category_id'] for ann in anns]
            anns_all = np.array(anns_all)

            # Filter for amount of uniif difference != 0:
            if difference == 0:
                sel_idx = np.ones((1, len(anns)), dtype=bool)
            else:
                sel_idx = np.ones((difference, len(anns)), dtype=bool)

            # Compute the original area of the objects to control for too massive occlusion, annotations are slightly inaccurate
            original_areas = np.zeros(len(anns))
            for ann_id in range(len(anns)):
                ann = anns[ann_id]
                mask = coco.annToMask(ann)
                original_areas[ann_id] = area(encode(mask))

            for d in range(sel_idx.shape[0]):  # Sliding window
                if difference == 0:
                    sel_idx[d, :] = True
                else:
                    # Filter out how many are left with this offset:
                    area_sel = np.zeros(len(anns_all))
                    for ann_id in range(len(anns)):
                        ann = anns[ann_id]
                        mask = coco.annToMask(ann)
                        if img['width'] > img['height']:
                            mask_sel = np.asfortranarray(mask[:, d:img['width'] - (difference - d)])
                        elif img['width'] < img['height']:
                            mask_sel = np.asfortranarray(mask[d:img['height'] - (difference - d), :])

                        area_sel[ann_id] = area(encode(mask_sel))
                        if area_sel[ann_id] < min_area_img:
                            sel_idx[d, ann_id] = False

            if difference != 0:
                best = np.where(np.array(np.sum(sel_idx, axis=1) == np.max(np.sum(sel_idx, axis=1))) == True)[0]
                if center_bias == True:
                    criterion = np.where(abs(best - difference / 2) == np.min(abs((best - difference / 2))))[0][0]
                else:
                    criterion = np.random.randint(len(best))
                d_sel.append(best[criterion])
            else:
                d_sel.append(0)

            if np.any(sel_idx[d_sel[-1], :] == True):
                anns_final = np.where(sel_idx[d_sel[-1], :] == True)[0]
                anns_plot = []
                annsIds_save = []
                if computeCoM == True:
                    CoM_save = []
                    # Determine the Centre of Mass
                    # select the chosen image square
                    selection = np.zeros((img['height'], img['width']), dtype=bool)
                    if img['width'] > img['height']:
                        selection[:, d_sel[-1]:img['width'] - ((img['width'] - img['height']) - d_sel[-1])] = True
                        dim_sel = img['height']
                        middle = [d_sel[-1] + dim_sel / 2, dim_sel / 2]
                    elif img['width'] < img['height']:
                        selection[d_sel[-1]:img['height'] - ((img['height'] - img['width']) - d_sel[-1]), :] = True
                        dim_sel = img['width']
                        middle = [dim_sel / 2, d_sel[-1] + dim_sel / 2]
                    else:
                        selection[:] = True
                        dim_sel = img['width']
                        middle = [dim_sel / 2, dim_sel / 2]
                targets = []
                counts = np.zeros((1,len(dataset['cat_coco_ids'])))
                y = np.zeros((1, len(dataset['cat_coco_ids'])))
                y_ann = []
                for ann_id in anns_final:
                    ann = anns[ann_id]
                    annsIds_save.append(ann['id'])

                    if computeCoM == True:

                        # Compute area
                        mask = coco.annToMask(ann)
                        mask_sel = mask[selection].reshape(dim_sel, dim_sel)
                        # target_area = area(encode(np.asfortranarray(mask_sel)))

                        m = mask_sel / np.sum(mask_sel[:])

                        # 1. obtain marginal distributions, 2. compute expected values
                        CoM_save.append(np.array([np.sum(np.sum(m, axis=0) * np.arange(m.shape[0])),
                                                  np.sum(
                                                      np.sum(m, axis=1) * np.arange(m.shape[1]))]) / dim_sel)


                    for h in range(len(dataset['cat_coco_ids'])): # This is quite ugly but a way to deal with the merged categories.
                        curr = False
                        if isinstance(dataset['cat_coco_ids'][h], list):
                            if ann['category_id'] in dataset['cat_coco_ids'][h]:
                                curr = True
                        else:
                            if dataset['cat_coco_ids'][h] == ann['category_id']:
                                curr = True
                        if curr == True:
                            targets.append(str(ann['category_id']) + ': ' + dataset['categories'][h])
                            counts[0, h] = counts[0, h] + 1
                            y_ann.append(np.zeros((1, len(dataset['cat_coco_ids']))))
                            y_ann[-1][0,h] = 1
                            y[0, h] = 1
                    anns_plot.append(ann)
                print(targets)

                if make_dataset == True:
                    I = io.imread(img['coco_url'])
                    if squaring == True:
                        if img['width'] > img['height']:
                            offset = [d_sel[-1], img['width'] - (difference - d_sel[-1])]
                            I_squared = I[:, offset[0]:offset[1]]

                        elif img['width'] < img['height']:
                            offset = [d_sel[-1], img['height'] - (difference - d_sel[-1])]
                            I_squared = I[offset[0]:offset[1], :]
                        else:
                            I_squared = I
                    else:
                        I_squared = I

                    io.imsave(
                        dataDir + 'images/' + dataset_name + '/' + dataType + '_' + targetType + '_' + spatial_constraint+'/' + img['file_name'],I_squared)

                    if 'x_ids' not in dataset:
                        dataset['x_ids'] = np.array(img['id'])
                        dataset['y'] = y
                        dataset['counts'] = counts
                        dataset['d_sel'] = d_sel
                        dataset['annIds'] = {}
                        dataset['annIds'][0] = annsIds_save
                        dataset['CoM'] = {}
                        dataset['CoM'][0] = CoM_save
                        dataset['y_ann']={}
                        dataset['y_ann'][0] = y_ann


                    else:
                        dataset['x_ids'] = np.append(dataset['x_ids'], img['id'])
                        dataset['y'] = np.append(dataset['y'], y, axis=0)
                        dataset['counts'] = np.append(dataset['counts'], counts, axis=0)
                        dataset['d_sel'] = d_sel
                        dataset['annIds'][len(dataset['x_ids']) - 1] = annsIds_save
                        dataset['CoM'][len(dataset['x_ids']) - 1] = CoM_save
                        dataset['y_ann'][len(dataset['x_ids']) - 1] = y_ann


                else:
                    print('No eligible targets for this image')
                    anns_plot = []

        elif targetType == 'single':
            for cat in dataset['cat_coco_ids']:
                annIds = coco.getAnnIds(catIds=cat, imgIds=img['id'], areaRng=[min_area_img, np.inf], iscrowd=None)

                if annIds != []:
                    anns = coco.loadAnns(annIds)

                    print('Found ' + str(len(anns)) + ' ' + str(cat) + '- annotations')

                    anns_all = [ann['category_id'] for ann in anns]
                    anns_all = np.array(anns_all)

                    # Filter for amount of uniif difference != 0:
                    if difference == 0:
                        sel_idx = np.ones((1, len(anns)), dtype=bool)
                    else:
                        sel_idx = np.ones((difference, len(anns)), dtype=bool)

                    # Compute the original area of the objects to control for too massive occlusion,
                    # annotations are slightly inaccurate
                    original_areas = np.zeros(len(anns))
                    for ann_id in range(len(anns)):
                        ann = anns[ann_id]
                        mask = coco.annToMask(ann)
                        original_areas[ann_id] = area(encode(mask))

                    for d in range(sel_idx.shape[0]):  # Sliding window
                        if difference == 0:
                            sel_idx[d, :] = True

                            masks_sel = []
                            for ann_id in range(len(anns)):
                                ann = anns[ann_id]
                                mask = coco.annToMask(ann)
                                masks_sel.append(mask)

                        else:
                            # Filter out how many are left with this offset:
                            area_sel = np.zeros(len(anns_all))
                            masks_sel = []
                            for ann_id in range(len(anns)):
                                ann = anns[ann_id]
                                mask = coco.annToMask(ann)
                                if img['width'] > img['height']:
                                    mask_sel = mask[:, d:img['width'] - (difference - d)]
                                elif img['width'] < img['height']:
                                    mask_sel = mask[d:img['height'] - (difference - d), :]
                                masks_sel.append(mask_sel)
                                area_sel[ann_id] = area(encode(np.asfortranarray(mask_sel)))
                                if area_sel[ann_id] < min_area_img:
                                    sel_idx[d, ann_id] = False

                        # Get rid of the duplicates
                        if np.sum(sel_idx[d,:])!=1:
                            sel_idx[d, :] = False
                        #anns_sel = anns_all[sel_idx[d, :]]
                        # This old piece of code can filter for uniqueness across multiple cats.
                        # This no longer neccessary since we are optimizing for a single cat at a time.
                        #duplicates = [len(np.where(h == anns_all[sel_idx[d, :]])[0]) for h in np.unique(anns_all[sel_idx[d, :]])]
                        #if np.any(np.array(duplicates) > 1):
                        #    for k in np.where(np.array(duplicates) > 1)[0]:
                        #        sel_idx[d, np.where(anns_all == np.unique(anns_all[sel_idx[d, :]])[k])[0]] = False
                        else:
                          # Check occlusion
                            if not img['width'] == img['height']:
                                for ann_id in range(len(anns)):
                                    if original_areas[ann_id] * 0.8 > area_sel[ann_id]:  # This is a check that not more than 80% of the object are cut-off
                                        sel_idx[d, ann_id] = False

                            if spatial_constraint == 'lateralized': # target object has to be either entirely on the left or on the rights side
                                # Determine whether lateralized
                                if img['width'] > img['height']:
                                    middle = np.mean([d, img['width'] - (difference - d)])
                                else:
                                    middle = np.mean([0, img['width']])

                                for sel in np.where(sel_idx[d, :] == True)[0]:
                                    ann = anns[sel]
                                    if (ann['bbox'][0] < middle) & (ann['bbox'][0] + ann['bbox'][2] < middle):
                                        sel_idx[d, sel] = True
                                    elif (ann['bbox'][0] > middle):
                                        sel_idx[d, sel] = True
                                    else:
                                        sel_idx[d, sel] = False

                            elif spatial_constraint == 'radial':
                                # Determine centre of the image
                                if img['width'] > img['height']:
                                    dim = img['height']
                                elif img['width'] < img['height']:
                                    dim = img['width']
                                else:
                                    dim = img['width']

                                radius = np.ceil(dim * 0.05)

                                # template = 0 * mask
                                [cx, cy] = np.meshgrid(
                                        np.linspace(-radius / 2 + 0.5, radius / 2 - 0.5, num=radius),
                                        np.linspace(-radius / 2 + 0.5, radius / 2 - 0.5, num=radius))
                                cz = np.sqrt(cx ** 2 + cy ** 2)  #
                                circle = cz < radius / 2
                                template = np.zeros((dim, dim),dtype=bool)
                                template[int(dim/2-radius/2):int(dim/2+radius/2), int(dim/2-radius/2):int(dim/2+radius/2)] = circle

                                for sel in np.where(sel_idx[d, :] == True)[0]:
                                    sel_idx[d, sel] = not np.any((template + masks_sel[sel]) > 1)

                    # Select the best windows
                    if difference != 0:
                        best = np.where(np.array(np.sum(sel_idx, axis=1) == np.max(np.sum(sel_idx, axis=1))) == True)[0]
                        if center_bias == True:
                            criterion = np.where(abs(best - difference / 2) == np.min(abs((best - difference / 2))))[0][0]
                        else: # random choice
                            criterion = np.random.randint(len(best))
                        d_sel_tmp= best[criterion]
                    else:
                        d_sel_tmp = 0

                    if np.any(sel_idx[d_sel_tmp, :] == True):
                        d_sel.append(d_sel_tmp)
                        anns_final = np.where(sel_idx[d_sel[-1], :] == True)[0]
                        anns_plot = []
                        annsIds_save = []
                        targets = []
                        y = np.zeros((1, len(dataset['cat_coco_ids'])))
                        for ann_id in anns_final:
                            ann = anns[ann_id]
                            annsIds_save.append(ann['id'])
                            for h in range(len(dataset['cat_coco_ids'])):  # This is quite ugly but a way to deal with the merged categories.
                                curr = False
                                if isinstance(dataset['cat_coco_ids'][h], list):
                                    if ann['category_id'] in dataset['cat_coco_ids'][h]:
                                        curr = True
                                else:
                                    if dataset['cat_coco_ids'][h] == ann['category_id']:
                                        curr = True

                                if curr == True:
                                    targets.append(str(ann['category_id']) + ': ' + dataset['categories'][h])
                                    y[0, h] = 1

                            anns_plot.append(ann)
                        print(targets)

                        if make_dataset == True:
                            I = io.imread(img['coco_url'])
                            if squaring == True:
                                if img['width'] > img['height']:
                                    offset = [d_sel[-1], img['width'] - (difference - d_sel[-1])]
                                    I_squared = I[:, offset[0]:offset[1]]

                                elif img['width'] < img['height']:
                                    offset = [d_sel[-1], img['height'] - (difference - d_sel[-1])]
                                    I_squared = I[offset[0]:offset[1], :]
                                else:
                                    I_squared = I
                            else:
                                I_squared = I

                            io.imsave(dataDir + 'images/' + dataset_name + '/' + dataType + '_' + targetType + '_' + spatial_constraint+ '/' + str(annsIds_save[0]) + '_' + img['file_name'] , I_squared)

                            if 'x_ids' not in dataset:
                                dataset['x_ids'] = [str(annsIds_save[0]) + '_' + img['file_name']]
                                dataset['img_ids'] = np.array(img['id'])
                                dataset['y'] = y
                                dataset['d_sel'] = d_sel
                                dataset['annIds'] = {}
                                dataset['annIds'] = annsIds_save

                            else:
                                dataset['x_ids'].append(str(annsIds_save[0]) + '_' + img['file_name'])
                                dataset['img_ids'] = np.append(dataset['img_ids'], img['id'])
                                dataset['y'] = np.append(dataset['y'], y, axis=0)
                                dataset['d_sel'] = d_sel
                                dataset['annIds'] = np.append(dataset['annIds'], annsIds_save)



        if (str(plotting).isdigit() == True) | (str(plotting) == 'all'):
            if anns_plot != []:
                plt.figure()
                I = io.imread(img['coco_url'])
                # plt.axis('off')
                plt.imshow(I)
                if squaring == True:
                    if img['width'] > img['height']:
                        offset = [d_sel[-1], img['width'] - (difference - d_sel[-1])]
                        middle = [np.mean(offset), img['height']/2]
                        plt.axvspan(0, offset[0], alpha=0.5, color='blue', label='_nolegend_')
                        plt.axvspan(offset[1], img['width'], alpha=0.5, color='blue', label='_nolegend_')
                    elif img['width'] < img['height']:
                        offset = [d_sel[-1], img['height'] - (difference - d_sel[-1])]
                        middle = [np.mean([0, img['width']]), np.mean(offset)]
                        plt.axhspan(0, offset[0], alpha=0.5, color='blue', label='_nolegend_')
                        plt.axhspan(offset[1], img['height'], alpha=0.5, color='blue', label='_nolegend_')

                    if (targetType == 'single') & (spatial_constraint== 'radial'):
                        ax = plt.gca()
                        ax.add_patch(plt.Circle(middle, radius/2, color='r', alpha=0.5))


                    elif (targetType == 'single') & (spatial_constraint== 'lateralized'):
                        plt.axvline(x=middle[0], alpha=0.5, color='red', label='_nolegend_')


                coco.showAnns(anns_plot)
                plt.title(targets)


             #plt.show()
                if str(plotting) == 'all':
                    plt.savefig(resultDir + dataset_name + '/AnnotatedImages/Example-' + str(img['id']) + '.png')
                else:
                    plt.savefig(resultDir + dataset_name + '/Example-' + str(img['id']) + '.png')

                plt.close()
        if saving == True:
            joblib.dump(dataset,
                        resultDir + dataset_name + '/' + str(dataType) + '_' + str(targetType)+ '_' + spatial_constraint + '.pickle')



#%%
mode = 'final_street'
min_area = 0.005
dataDir='/mnt/Googolplex/coco/'
resultDir = '/mnt/Googolplex/PycharmProjects/ASN_mac/COCO/Dataset/'

squaring = True
saving = True
make_dataset = True
#%%
if mode == 'final_street':
    dataTypes = ['val2017', 'train2017']
    dataset_name = '1_street'
    stuff_cats = ['road']  # , 'pavement']
    object_cats = ['person','bicycle','car','motorcycle','bus','truck', 'fire hydrant',
                'stop sign']

    for dataType in dataTypes:
        if dataType == 'val2017':
            makeDataset(dataset_name, dataDir, resultDir, dataType, object_cats, stuff_cats=stuff_cats, targetType='single',
                        saving=saving, make_dataset=make_dataset, computeCoM=True)

        makeDataset(dataset_name, dataDir, resultDir, dataType, object_cats, stuff_cats=stuff_cats, targetType='multi',
                     saving=saving, make_dataset=make_dataset,computeCoM=True)

elif mode == 'final_food':
    dataTypes = ['train2017','val2017']
    dataset_name = '2_food'
    stuff_cats = ['table', 'cloth', 'food-other', 'vegetable', 'salad', 'fruit', 'napkin']
    object_cats = ['bottle','wine glass','cup','fork','knife','spoon','bowl','sandwich',
                'carrot','cake']
    merged_cats = {}
    merged_cats['cutlery'] = ['fork','knife','spoon']
    plotting = 'all'
    for dataType in dataTypes:
        if dataType == 'val2017':
            makeDataset(dataset_name, dataDir,resultDir, dataType, object_cats, stuff_cats=stuff_cats, merged_cats= merged_cats,
                        plotting=plotting, targetType='single',saving=saving, make_dataset=make_dataset)
        makeDataset(dataset_name, dataDir, resultDir,dataType, object_cats, stuff_cats=stuff_cats, merged_cats= merged_cats,
                    targetType='multi', saving=saving, make_dataset=make_dataset, computeCoM=True)


