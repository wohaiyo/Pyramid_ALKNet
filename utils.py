import cv2
import config as cfg
import numpy as np
import os

def eval_img2(gt, pred):
    cls_acc = []
    TP_num = []
    pos_all_total = []


    for i in range(1, cfg.NUM_OF_CLASSESS):   # ignore 0
        gt_i = np.zeros(gt.shape, np.int)
        pred_i = np.zeros(gt.shape, np.int)
        zero = np.zeros(gt.shape, np.int)

        gt_i[gt == i] = 1
        pred_i[pred == i] = 1

        cls_i = gt_i.copy()
        cls_i[gt != i] = -1
        zero[cls_i == pred_i] = 1

        TP = np.sum(zero)

        pos_all = np.sum(gt_i)

        if pos_all == 0:
            cls_acc.append(-1)
        else:
            cls_acc.append(TP / pos_all)
        TP_num.append(TP)
        pos_all_total.append(pos_all)

    return cls_acc, sum(TP_num) / sum(pos_all_total),  TP_num, pos_all_total

def eval_fscore(gt, pred):
    TPs = []
    FPs = []
    FNs = []

    for i in range(1, cfg.NUM_OF_CLASSESS):   # ignore 0
        gt_i = np.zeros(gt.shape, np.int)
        pred_i = np.zeros(gt.shape, np.int)
        zero_tp = np.zeros(gt.shape, np.int)

        # TP
        gt_i[gt == i] = 1
        pred_i[pred == i] = 1

        cls_i = gt_i.copy()
        cls_i[gt != i] = -1
        zero_tp[cls_i == pred_i] = 1

        TP = np.sum(zero_tp)

        # FP
        gt_i[gt != i] = 1
        gt_i[gt == i] = 0
        gt_i[gt == 0] = 0

        pred_i[pred == i] = 1
        pred_i[pred != i] = 0

        FP = np.sum(gt_i * pred_i)

        # FN
        gt_i[gt != i] = 0
        gt_i[gt == i] = 1
        gt_i[gt == 0] = 0

        pred_i[pred == i] = 0
        pred_i[pred != i] = 1

        FN = np.sum(gt_i * pred_i)

        TPs.append(TP)
        FPs.append(FP)
        FNs.append(FN)

    return TPs, FPs, FNs

def eval_img(gt, pred):
    height = gt.shape[0]
    width = gt.shape[1]

    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    # F1 = 2 * Precision * Recall / (Precision + Recall)

    cls_acc = []
    TP_num = []
    pos_all_total = []

    for num in range(1, cfg.NUM_OF_CLASSESS):   # ignore 0
        TP = 0
        FN = 0
        pos_all = 0
        for h in range(height):
            for w in range(width):
                if gt[h][w][0] == num:
                    pos_all += 1
                    if gt[h][w][0] == pred[h][w][0]:
                        TP += 1
                    else:
                        FN += 1

        if pos_all == 0:                # no class label
            cls_acc.append(-1.)
        else:
            cls_acc.append(TP / pos_all)

        TP_num.append(TP)
        pos_all_total.append(pos_all)

    return cls_acc, sum(TP_num) / sum(pos_all_total),  TP_num, pos_all_total

def pred_vision(pred, name, dataset):  # pred （h, w， 1)
    pred = np.array(pred)
    height = pred.shape[0]
    width = pred.shape[1]

    pred_new = np.zeros([height, width, 3], dtype=np.uint8)
    pred_B = np.zeros([height, width, 1], dtype=np.uint8)
    pred_G = np.zeros([height, width, 1], dtype=np.uint8)
    pred_R = np.zeros([height, width, 1], dtype=np.uint8)

    if 'etrims' in dataset:
        label_color = [[0, 0, 0, 0],
                       [1, 0, 0, 128],
                       [2, 128, 0, 128],
                       [3, 0, 128, 128],
                       [4, 128, 128, 128],
                       [5, 0, 64, 128],
                       [6, 128, 128, 0],
                       [7, 0, 128, 0],
                       [8, 128, 0, 0],
                       ]
        label_color = np.array(label_color, np.int)
        for i in range(label_color.shape[0]):
            pred_B[pred == label_color[i][0]] = label_color[i][1]
            pred_G[pred == label_color[i][0]] = label_color[i][2]
            pred_R[pred == label_color[i][0]] = label_color[i][3]

    elif 'cmp' in dataset:      # ['Background','Door', 'Shop', 'Balcony', 'Window', 'Wall']    # (0,0,0) ()
        label_color = [[0, 0, 0, 0],
                       [1, 255, 170, 0],
                       [2, 0, 0, 170],
                       [3, 85, 255, 170],
                       [4, 255, 85, 0],
                       [5, 255, 0, 0]
                       ]
        label_color = np.array(label_color, np.int)
        for i in range(label_color.shape[0]):
            pred_B[pred == label_color[i][0]] = label_color[i][1]
            pred_G[pred == label_color[i][0]] = label_color[i][2]
            pred_R[pred == label_color[i][0]] = label_color[i][3]

    elif 'art' in dataset or 'Art' in dataset:
        label_color = [[0, 0, 0, 0],
                       [1, 0, 128, 255],
                       [2, 0, 255, 0],
                       [3, 255, 0, 128],
                       [4, 0, 0, 255],
                       [5, 0, 255, 255],
                       [6, 255, 255, 128],
                       [7, 255, 0, 0],
                       ]
        label_color = np.array(label_color, np.int)
        for i in range(label_color.shape[0]):
            pred_B[pred == label_color[i][0]] = label_color[i][1]
            pred_G[pred == label_color[i][0]] = label_color[i][2]
            pred_R[pred == label_color[i][0]] = label_color[i][3]
    elif 'camvid' in dataset:
        label_color = [[0, 0, 0, 0],
                                [1, 128, 128, 128],  # R G B
                                [2, 128, 0, 0],
                                [3, 192, 192, 192],
                                [4, 128, 64, 128],
                                [5, 60, 40, 222],
                                [6, 128, 128, 0],
                                [7, 192, 128, 128],
                                [8, 64, 64, 128],
                                [9, 64, 0, 128],
                                [10, 64, 64, 0],
                                [11, 0, 128, 192]]

        label_color = np.array(label_color, np.int)
        for i in range(label_color.shape[0]):
            pred_B[pred == label_color[i][0]] = label_color[i][3]
            pred_G[pred == label_color[i][0]] = label_color[i][2]
            pred_R[pred == label_color[i][0]] = label_color[i][1]

    elif 'Rue' in dataset or 'Monge' in dataset:
        label_color = [[0, 0, 0, 0],
                       [1, 0, 0, 255],
                       [2, 0, 255, 255],
                       [3, 255, 0, 128],
                       [4, 0, 128, 255],
                       [5, 255, 0, 0],
                       [6, 255, 255, 128],
                       [7, 0, 255, 0],
                       ]
        label_color = np.array(label_color, np.int)
        for i in range(label_color.shape[0]):
            pred_B[pred == label_color[i][0]] = label_color[i][1]
            pred_G[pred == label_color[i][0]] = label_color[i][2]
            pred_R[pred == label_color[i][0]] = label_color[i][3]

    else:                                               # ECP
        label_color = [[0, 0, 0, 0],
                       [1, 0, 0, 255],
                       [2, 0, 255, 255],
                       [3, 255, 0, 128],
                       [4, 0, 128, 255],
                       [5, 255, 0, 0],
                       [6, 128, 128, 128],
                       [7, 255, 255, 128],
                       [8, 0, 255, 0],
                       ]
        label_color = np.array(label_color, np.int)
        for i in range(label_color.shape[0]):
            pred_B[pred == label_color[i][0]] = label_color[i][1]
            pred_G[pred == label_color[i][0]] = label_color[i][2]
            pred_R[pred == label_color[i][0]] = label_color[i][3]

    pred_new = np.concatenate([pred_B, pred_G, pred_R], 2)
    save_name = cfg.save_dir + 'output/' + name + '.png'
    cv2.imwrite(save_name, pred_new)
    print(save_name + ' is saved.')

def pred_vision_path(pred, save_path, dataset):  # pred （h, w， 1)
    pred = np.array(pred)
    height = pred.shape[0]
    width = pred.shape[1]

    pred_new = np.zeros([height, width, 3], dtype=np.uint8)
    pred_B = np.zeros([height, width, 1], dtype=np.uint8)
    pred_G = np.zeros([height, width, 1], dtype=np.uint8)
    pred_R = np.zeros([height, width, 1], dtype=np.uint8)

    if 'etrims' in dataset:
        label_color = [[0, 0, 0, 0],
                       [1, 0, 0, 128],
                       [2, 128, 0, 128],
                       [3, 0, 128, 128],
                       [4, 128, 128, 128],
                       [5, 0, 64, 128],
                       [6, 128, 128, 0],
                       [7, 0, 128, 0],
                       [8, 128, 0, 0],
                       ]
        label_color = np.array(label_color, np.int)
        for i in range(label_color.shape[0]):
            pred_B[pred == label_color[i][0]] = label_color[i][1]
            pred_G[pred == label_color[i][0]] = label_color[i][2]
            pred_R[pred == label_color[i][0]] = label_color[i][3]

    elif 'cmp' in dataset:      # ['Background','Door', 'Shop', 'Balcony', 'Window', 'Wall']    # (0,0,0) ()
        label_color = [[0, 0, 0, 0],
                       [1, 255, 170, 0],
                       [2, 0, 0, 170],
                       [3, 85, 255, 170],
                       [4, 255, 85, 0],
                       [5, 255, 0, 0]
                       ]
        label_color = np.array(label_color, np.int)
        for i in range(label_color.shape[0]):
            pred_B[pred == label_color[i][0]] = label_color[i][1]
            pred_G[pred == label_color[i][0]] = label_color[i][2]
            pred_R[pred == label_color[i][0]] = label_color[i][3]

    elif 'art' in dataset or 'Art' in dataset:
        label_color = [[0, 0, 0, 0],
                       [1, 0, 128, 255],
                       [2, 0, 255, 0],
                       [3, 255, 0, 128],
                       [4, 0, 0, 255],
                       [5, 0, 255, 255],
                       [6, 255, 255, 128],
                       [7, 255, 0, 0],
                       ]
        label_color = np.array(label_color, np.int)
        for i in range(label_color.shape[0]):
            pred_B[pred == label_color[i][0]] = label_color[i][1]
            pred_G[pred == label_color[i][0]] = label_color[i][2]
            pred_R[pred == label_color[i][0]] = label_color[i][3]
    elif 'camvid' in dataset:
        label_color = [[0, 0, 0, 0],
                                [1, 128, 128, 128],  # R G B
                                [2, 128, 0, 0],
                                [3, 192, 192, 192],
                                [4, 128, 64, 128],
                                [5, 60, 40, 222],
                                [6, 128, 128, 0],
                                [7, 192, 128, 128],
                                [8, 64, 64, 128],
                                [9, 64, 0, 128],
                                [10, 64, 64, 0],
                                [11, 0, 128, 192]]

        label_color = np.array(label_color, np.int)
        for i in range(label_color.shape[0]):
            pred_B[pred == label_color[i][0]] = label_color[i][3]
            pred_G[pred == label_color[i][0]] = label_color[i][2]
            pred_R[pred == label_color[i][0]] = label_color[i][1]

    elif 'Rue' in dataset or 'Monge' in dataset:
        label_color = [[0, 0, 0, 0],
                       [1, 0, 0, 255],
                       [2, 0, 255, 255],
                       [3, 255, 0, 128],
                       [4, 0, 128, 255],
                       [5, 255, 0, 0],
                       [6, 255, 255, 128],
                       [7, 0, 255, 0],
                       ]
        label_color = np.array(label_color, np.int)
        for i in range(label_color.shape[0]):
            pred_B[pred == label_color[i][0]] = label_color[i][1]
            pred_G[pred == label_color[i][0]] = label_color[i][2]
            pred_R[pred == label_color[i][0]] = label_color[i][3]

    else:                                               # ECP
        label_color = [[0, 0, 0, 0],
                       [1, 0, 0, 255],
                       [2, 0, 255, 255],
                       [3, 255, 0, 128],
                       [4, 0, 128, 255],
                       [5, 255, 0, 0],
                       [6, 128, 128, 128],
                       [7, 255, 255, 128],
                       [8, 0, 255, 0],
                       ]
        label_color = np.array(label_color, np.int)
        for i in range(label_color.shape[0]):
            pred_B[pred == label_color[i][0]] = label_color[i][1]
            pred_G[pred == label_color[i][0]] = label_color[i][2]
            pred_R[pred == label_color[i][0]] = label_color[i][3]

    pred_new = np.concatenate([pred_B, pred_G, pred_R], 2)
    cv2.imwrite(save_path, pred_new)
    print(save_path + ' is saved.')


def pred_vision2(pred, name, dataset):       # pred （h, w， 1)
    pred = np.array(pred)
    height = pred.shape[0]
    width = pred.shape[1]

    pred_new = np.zeros([height, width, 3], dtype=np.uint8)
    if 'etrims' in dataset:
        for y in range(height):
            for x in range(width):      # color(B, G, R)
                if (pred[y][x] == 1):  # Building
                    pred_new[y][x] = np.array([0, 0, 128])
                elif (pred[y][x] == 2):  # Car
                    pred_new[y][x] = np.array([128, 0, 128])
                elif (pred[y][x] == 3):  # Door
                    pred_new[y][x] = np.array([0, 128, 128])
                elif (pred[y][x] == 4):  # Pavement
                    pred_new[y][x] = np.array([128, 128, 128])
                elif (pred[y][x] == 5):  # Road
                    pred_new[y][x] = np.array([0, 64, 128])
                elif (pred[y][x] == 6):  # Sky
                    pred_new[y][x] = np.array([128, 128, 0])
                elif (pred[y][x] == 7):  # Vegetation
                    pred_new[y][x] = np.array([0, 128, 0])
                elif (pred[y][x] == 8):  # Window
                    pred_new[y][x] = np.array([128, 0, 0])
                else:
                    pred_new[y][x] = np.array([0, 0, 0])
    elif 'cmp' in dataset:      # ['Background','Door', 'Shop', 'Balcony', 'Window', 'Wall']    # (0,0,0) ()
        for y in range(height):
            for x in range(width):      # color(B, G, R)
                if (pred[y][x] == 1):  # Door
                    pred_new[y][x] = np.array([255, 170, 0])
                elif (pred[y][x] == 2):  # Shop
                    pred_new[y][x] = np.array([0, 0, 170])
                elif (pred[y][x] == 3):  # Balcony
                    pred_new[y][x] = np.array([85, 255, 170])
                elif (pred[y][x] == 4):  # Window
                    pred_new[y][x] = np.array([255, 85, 0])
                elif (pred[y][x] == 5):  # Wall
                    pred_new[y][x] = np.array([255, 0, 0])
                else:
                    pred_new[y][x] = np.array([0, 0, 0])
    elif 'art' in dataset or 'Art' in dataset:
        for y in range(height):
            for x in range(width):  # color(B, G, R)
                if (pred[y][x] == 1):  # Door
                    pred_new[y][x] = np.array([0, 128, 255])
                elif (pred[y][x] == 2):  # Shop
                    pred_new[y][x] = np.array([0, 255, 0])
                elif (pred[y][x] == 3):  # Balcony
                    pred_new[y][x] = np.array([255, 0, 128])
                elif (pred[y][x] == 4):  # Window
                    pred_new[y][x] = np.array([0, 0, 255])
                elif (pred[y][x] == 5):  # Wall
                    pred_new[y][x] = np.array([0, 255, 255])
                elif (pred[y][x] == 6):  # Sky
                    pred_new[y][x] = np.array([255, 255, 128])
                elif (pred[y][x] == 7):  # Roof
                    pred_new[y][x] = np.array([255, 0, 0])
                else:
                    pred_new[y][x] = np.array([0, 0, 0])
    else:
        for y in range(height):
            for x in range(width):  # color(B, G, R)
                if (pred[y][x] == 1):  # Window
                    pred_new[y][x] = np.array([0, 0, 255])
                elif (pred[y][x] == 2):  # Wall
                    pred_new[y][x] = np.array([0, 255, 255])
                elif (pred[y][x] == 3):  # Balcony
                    pred_new[y][x] = np.array([255, 0, 128])
                elif (pred[y][x] == 4):  # Door
                    pred_new[y][x] = np.array([0, 128, 255])
                elif (pred[y][x] == 5):  # Roof
                    pred_new[y][x] = np.array([255, 0, 0])
                elif (pred[y][x] == 6):  # Chimney
                    pred_new[y][x] = np.array([128, 128, 128])
                elif (pred[y][x] == 7):  # Sky
                    pred_new[y][x] = np.array([255, 255, 128])
                elif (pred[y][x] == 8):  # Shop
                    pred_new[y][x] = np.array([0, 255, 0])
                else:
                    pred_new[y][x] = np.array([0, 0, 0])
    save_name = cfg.save_dir + 'output/' + name + '.png'
    cv2.imwrite(save_name, pred_new)
    print(save_name + ' is saved.')

def PCA_compress(feature, channel=1):
    '''
    Compress CNN feature to 3 dimensions and normlization
    :param feature:
    :param channel:
    :return:
    '''
    import numpy as np
    from sklearn.decomposition import PCA
    h = feature.shape[1]
    w = feature.shape[2]
    c = feature.shape[3]
    feature = np.reshape(np.squeeze(feature, 0),[-1, c])
    pca = PCA(n_components=channel)
    pca.fit(feature)
    newX = pca.fit_transform(feature)
    newX = np.reshape(newX, [h, w, channel])

    # Normlizaiton
    max_v = np.max(newX)
    min_v = np.min(newX)
    newX_norm = (newX - min_v) / (max_v - min_v)
    return [newX_norm]

def t_sne_compress(feature, channel=3):
    '''
    Compress CNN feature to 3 dimensions and normlization
    :param feature:
    :param channel:
    :return:
    '''
    import numpy as np
    from sklearn import manifold
    h = feature.shape[1]
    w = feature.shape[2]
    c = feature.shape[3]
    feature = np.reshape(np.squeeze(feature, 0),[-1, c])
    tsne = manifold.TSNE(n_components=channel, init='pca', random_state=501)
    print('t_sne fitting...')
    newX = tsne.fit_transform(feature)
    newX = np.reshape(newX, [h, w, channel])

    # Normlizaiton
    max_v = np.max(newX)
    min_v = np.min(newX)
    newX_norm = (newX - min_v) / (max_v - min_v)
    return [newX_norm]

def visual_2d(X, y, name, class_num = cfg.NUM_OF_CLASSESS):
    '''
    visualization of 2d
    :param X: coordinate
    :param y: label
    :return:
    '''
    print('Drawing 2d coor')
    X = np.resize(X, [-1, 2])
    y = np.squeeze(np.resize(y, [-1, 1]), 1)
    import matplotlib.pyplot as plt
    colors = ['#FFF000', '#808080', '#FFC0CB', '#EE00EE', '#008000', '#CD950C', '#9F79EE', '#EEB422', '#F08080']


    fig = plt.figure(figsize=(8, 8))
    area = np.pi * (3)**2
    for i in range(X.shape[0]):
        plt.scatter(X[i, 0], X[i, 1], area, c=colors[y[i]], marker='.', alpha=0.8)
    # plt.show()

    fig.savefig(name + '_feature2d.jpg')

