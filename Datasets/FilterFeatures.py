
from asn.utils import load_pickle, save_pickle
import pandas as pd
import numpy as np

stem_dir = '/mnt/Googolplex/PycharmProjects/ASN_mac/COCO/Dataset/'
selection_counts = pd.DataFrame()
case = 1
targetType = 'single'
spatial_constraint = 'radial'
if case == 1:
    criteria = pd.DataFrame({'Bounds':['lower','upper'],
                             'sceneComplexity_SC':[0,1.2],
                            'targetSalienceSum': [0.04, np.inf]})
elif case == 2:
    criteria = pd.DataFrame({'Bounds': ['lower', 'upper'],
                             'sceneComplexity_SC': [0, 1.15],
                             'targetSalienceSum': [0.045, np.inf]})

for d in ['1_street']:#['1_street','2_food']:
    path = stem_dir + d + '/'
    features = load_pickle(path+'Features_' + targetType + '_' + spatial_constraint + '.pickle')

    data = pd.read_pickle(path + 'FeatureDataframe_' + targetType + '_' + spatial_constraint)
    selection_cats = list(features['categories'].copy())
    if len(selection_cats) < 10:
        for p in range(10-len(selection_cats)):
            selection_cats.append([])
    selection_cats.append('all')
    selection_counts[d + '_categories'] = selection_cats
    selection_counts[d + '_counts'] = np.zeros(len(selection_cats))
    selection_counts[d + '_counts'][len(selection_cats)-1] = len(features['x_ids'])
    for c in range(len(features['categories'])):
        cat = features['cat_coco_ids'][c]
        selection_counts[d + '_counts'][c] = np.sum(features['y'][:, c])

    selection = pd.DataFrame(np.ones((data.shape[0],criteria.shape[1]-1),dtype=bool), columns=[criteria.columns[1:]])

    for c in selection.columns:
        print(c[0]) # select outliers
        selection[c][(data[c[0]] < criteria[c[0]][0])|(data[c[0]]> criteria[c[0]][1])] = False

    #selection['selected'] = selection.select_dtypes(include=['bool']).sum(axis=1) == criteria.shape[1]-1

    data['selection_' + str(case)] = selection.select_dtypes(include=['bool']).sum(axis=1) == criteria.shape[1]-1
    data.to_pickle(path + 'FeatureDataframe_' + targetType + '_' + spatial_constraint)

    features['selection_'+ str(case)] =data['selection_'+ str(case)].values
    save_pickle(features, path+'Features_' + targetType + '_' + spatial_constraint)

    selection_counts[d + '_selected_counts'] = np.zeros(len(selection_cats))
    selection_counts[d + '_selected_counts'][len(selection_cats)-1] = len(np.unique(data['img_ids'][data['selection_' + str(case)]]))
    for c in range(len(features['categories'])):
        cat = features['categories'][c]
        selection_counts[d + '_selected_counts'][c] = len(np.unique(data['img_ids'][(data['selection_' + str(case)]) & (data['targetCategory'] == cat)]))

    selection_counts.to_csv(stem_dir+'SelectionCounts_' + targetType + '_' + spatial_constraint + '.csv')







