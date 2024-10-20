import json, pickle
import os
from sklearn.metrics import accuracy_score
import math
import torch
import torch.nn.functional as F
from openpyxl import load_workbook
from openpyxl.utils.cell import coordinate_from_string, column_index_from_string
from openpyxl import Workbook
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2
import random
from datetime import datetime

def load_eval_file(file_path):
    if os.path.exists(file_path):
        result_dict = json.load(open(file_path))
    return result_dict


def sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y

def extract_centers(heatmap, obj_name, thresh=0.5, radius=2):
    device = heatmap.device
    batch, c, h, w = heatmap.shape
    all_centers = []

    # print('heat map shape: {}'.format(heatmap.shape))
    # print(heatmap[1, 0, :, :])

    if obj_name == 'motorcycle' or obj_name == 'bicycle':
        thresh = 0.6

    for b in range(batch):
        batch_centers = []
        for ch in range(c):
            heatmap_bch = heatmap[b, ch, :, :]

            # print(heatmap_bch.shape)
            hm_np = heatmap.numpy()
            hm = (hm_np * 255).astype('uint8')

            # Apply threshold to the heatmap
            keep = heatmap_bch > thresh
            heatmap_bch = heatmap_bch * keep.float()

            # Suppress non-local maxima
            heatmap_max = F.max_pool2d(heatmap_bch.unsqueeze(0).unsqueeze(0), (2 * radius + 1, 2 * radius + 1), stride=1, padding=radius)
            keep = (heatmap_bch == heatmap_max.squeeze(0).squeeze(0))
            heatmap_bch = heatmap_bch * keep.float()

            # Get indices of the center points
            y_idx, x_idx = torch.nonzero(heatmap_bch, as_tuple=True)

            for i in range(len(y_idx)):
                batch_centers.append((x_idx[i].item(), y_idx[i].item()))

        all_centers.append(batch_centers)

    # Convert list of centers to tensor
    centers_tensor = [torch.tensor(centers, dtype=torch.float32, device=device) for centers in all_centers]
    max_len = max(len(centers) for centers in centers_tensor)
    padded_centers = torch.zeros(batch, max_len, 2, device=device)
    for i, centers in enumerate(centers_tensor):
        if len(centers) > 0:
            padded_centers[i, :len(centers), :] = centers

    num_centers = torch.tensor([len(centers) for centers in all_centers], dtype=torch.float32, device=device)

    # print('num_centers: {}'.format(num_centers))
    # print('padded centers: ')
    # print(padded_centers)

    return num_centers, padded_centers

def extract_centers_dynamic(heatmap, thresh=0.5, radius=2):
    device = heatmap.device
    batch, c, h, w = heatmap.shape
    all_centers = []

    # print('heat map shape: {}'.format(heatmap.shape))
    # print(heatmap[1, 0, :, :])
    for b in range(batch):
        batch_centers = []
        for ch in range(c):
            heatmap_bch = heatmap[b, ch, :, :]

            # print(heatmap_bch.shape)
            hm_np = heatmap.numpy()
            hm = (hm_np * 255).astype('uint8')

            ret, otsu_thresh = cv2.threshold(hm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # ret is the treshold
            # print(ret/255 + 0.1)
            if ret == 0:
                dynamic_thresh = 1
            else:
                dynamic_thresh = ret/255 + thresh
            # print('thresh: {} '.format(dynamic_thresh))

            # Apply threshold to the heatmap
            keep = heatmap_bch > dynamic_thresh
            heatmap_bch = heatmap_bch * keep.float()

            # Suppress non-local maxima
            heatmap_max = F.max_pool2d(heatmap_bch.unsqueeze(0).unsqueeze(0), (2 * radius + 1, 2 * radius + 1), stride=1, padding=radius)
            keep = (heatmap_bch == heatmap_max.squeeze(0).squeeze(0))
            heatmap_bch = heatmap_bch * keep.float()

            # Get indices of the center points
            y_idx, x_idx = torch.nonzero(heatmap_bch, as_tuple=True)

            for i in range(len(y_idx)):
                batch_centers.append((x_idx[i].item(), y_idx[i].item()))

        all_centers.append(batch_centers)

    # Convert list of centers to tensor
    centers_tensor = [torch.tensor(centers, dtype=torch.float32, device=device) for centers in all_centers]
    max_len = max(len(centers) for centers in centers_tensor)
    padded_centers = torch.zeros(batch, max_len, 2, device=device)
    for i, centers in enumerate(centers_tensor):
        if len(centers) > 0:
            padded_centers[i, :len(centers), :] = centers

    num_centers = torch.tensor([len(centers) for centers in all_centers], dtype=torch.float32, device=device)

    return num_centers, padded_centers

def extract_centers_dynamic2(heatmap, obj_name, thresh=0.5, radius=2):
    device = heatmap.device
    batch, c, h, w = heatmap.shape
    all_centers = []

    for b in range(batch):
        batch_centers = []
        for ch in range(c):
            heatmap_bch = heatmap[b, ch, :, :]

            # print(heatmap_bch.shape)
            hm_np = heatmap.numpy()
            hm = (hm_np * 255).astype('uint8')

            ret, otsu_thresh = cv2.threshold(hm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # ret is the treshold
            # print(ret/255 + 0.1)

            if obj_name == 'motorcycle' or obj_name == 'bicycle':
                thresh = 0.12
            if ret == 0:
                dynamic_thresh = 1
            else:
                dynamic_thresh = ret/255 + thresh
            # print('thresh: {} '.format(dynamic_thresh))

            # Apply threshold to the heatmap
            keep = heatmap_bch > dynamic_thresh
            heatmap_bch = heatmap_bch * keep.float()

            # Suppress non-local maxima
            heatmap_max = F.max_pool2d(heatmap_bch.unsqueeze(0).unsqueeze(0), (2 * radius + 1, 2 * radius + 1), stride=1, padding=radius)
            keep = (heatmap_bch == heatmap_max.squeeze(0).squeeze(0))
            heatmap_bch = heatmap_bch * keep.float()

            # Get indices of the center points
            y_idx, x_idx = torch.nonzero(heatmap_bch, as_tuple=True)

            for i in range(len(y_idx)):
                batch_centers.append((x_idx[i].item(), y_idx[i].item()))

        all_centers.append(batch_centers)

    # Convert list of centers to tensor
    centers_tensor = [torch.tensor(centers, dtype=torch.float32, device=device) for centers in all_centers]
    max_len = max(len(centers) for centers in centers_tensor)
    padded_centers = torch.zeros(batch, max_len, 2, device=device)
    for i, centers in enumerate(centers_tensor):
        if len(centers) > 0:
            padded_centers[i, :len(centers), :] = centers

    num_centers = torch.tensor([len(centers) for centers in all_centers], dtype=torch.float32, device=device)


    return num_centers, padded_centers


def hm(thresh=0.5, radius=2):
    print('thresh: {}, radius: {}'.format(thresh, radius))
    objs = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    gt_path = 'gt_list.txt'
    if os.path.exists(gt_path):
            gt_list = json.load(open(gt_path))

    result_dict = {}
    for i in range(15050): #15050
        obj_idx = i % 10
        file_path = '/data/hms/{}.txt'.format(i)
        if os.path.exists(file_path):
            hm= json.load(open(file_path))
            hm= torch.tensor(hm, dtype=torch.float32)
            hm = sigmoid(hm)
            num_centers, padded_centers = extract_centers(hm, objs[obj_idx], thresh, radius=radius)

            num_centers = num_centers.tolist()
            gt = gt_list[i]

        obj = objs[obj_idx]
        if obj in result_dict.keys():
            result_dict[obj]['gt_count'].extend(gt)
            result_dict[obj]['center_count'].extend(num_centers)
        else:
            result_dict[obj] = {'gt_count': gt, 'center_count': num_centers}


        if i % 2000 == 0:
            print(i)

    file = 'result_output/result_center.txt'
    with open(file, 'w') as f:
        json.dump(result_dict, f)
    return result_dict
def remove_duplicate_centers(tensor, radius=3):
    # Get the batch size and dimensions of each feature map
    batch_size, height, width = tensor.shape

    # Process each batch independently
    for b in range(batch_size):
        for i in range(height):
            for j in range(width):
                # If the current center is 1, process its neighborhood
                if tensor[b, i, j] == 1:
                    # Calculate the neighborhood bounds
                    min_x = max(0, i - radius)
                    max_x = min(height, i + radius + 1)
                    min_y = max(0, j - radius)
                    max_y = min(width, j + radius + 1)

                    # Set the surrounding centers within the radius to 0, except the center itself
                    for x in range(min_x, max_x):
                        for y in range(min_y, max_y):
                            # Check if the point is within the circle of given radius
                            if (x - i)**2 + (y - j)**2 <= radius**2:
                                if x != i or y != j:
                                    tensor[b, x, y] = 0
    return tensor

def hm_partition_overlap(thresh=0.05, radius=1, radius_duplicate = 2):
    print('thresh: {}, radius: {}, radius duplicate: {}'.format(thresh, radius, radius_duplicate))
    objs = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    gt_path = 'gt_list.txt'
    if os.path.exists(gt_path):
            gt_list = json.load(open(gt_path))

    result_dict = {}
    expanding_rate = 0.3
    expanding = math.floor(64 * expanding_rate)
    # ranges = [[0, 64+expanding, 0, 64+expanding],[0, 64+expanding, 64-expanding, 128],[64-expanding, 128, 0, 64+expanding],[64-expanding, 128, 64-expanding, 128]]
    ranges = [[0, 64 + expanding, 0, 128], [64-expanding, 128, 0, 128]]

    # expanding = math.floor(42 * expanding_rate)
    # ranges = [[0, 42+expanding, 0, 42+expanding], [42-expanding, 84+expanding, 0, 42+expanding], [84-expanding, 128, 0, 42+expanding], [0, 42+expanding, 42-expanding, 84+expanding], [42-expanding, 84+ expanding, 42-expanding, 84+expanding], [84-expanding, 128, 42-expanding, 84+expanding], [0, 42+expanding, 84-expanding, 128], [42-expanding, 84+expanding, 84-expanding, 128], [84-expanding, 128, 84-expanding, 128]]

    start_time = datetime.now()
    for i in range(15050): #150
        obj_idx = i % 10
        file_path = '/data/hm_partition2_overlap0.3/{}.txt'.format(i)
        # file_path = '/data1/hms_partition_overlap_2/{}.txt'.format(i)
        # file_path = '/data1/hms_partition_overlap_0.3/{}.txt'.format(i)
        if os.path.exists(file_path):
            hms= json.load(open(file_path))
            # partition number of hms
            maps = np.zeros((4, 128, 128)) # batch size
            for hm_idx, hm in enumerate(hms):
                coord_range = ranges[hm_idx]
                hm= torch.tensor(hm, dtype=torch.float32)
                hm = sigmoid(hm)
                num_centers, padded_centers = extract_centers_dynamic2(hm, objs[obj_idx], thresh, radius=radius)

                for batch_idx, batch_centers in enumerate(padded_centers):
                    for center in batch_centers:
                        y = int(center[0])
                        x = int(center[1])
                        if x != 0 and y != 0:
                            maps[batch_idx, coord_range[2] + y, coord_range[0] + x] = 1

            final_maps = remove_duplicate_centers(maps, radius_duplicate)

            num_centers = np.sum(final_maps, axis=(1, 2))

            #Print the sums for each batch
            # print("Sum of each batch:", num_centers)

            gt = gt_list[i]

            obj = objs[obj_idx]
            if obj in result_dict.keys():
                result_dict[obj]['gt_count'].extend(gt)
                result_dict[obj]['center_count'].extend(num_centers.tolist())
            else:
                result_dict[obj] = {'gt_count': gt, 'center_count': num_centers.tolist()}
        else:
            print('not exist')

        # if obj == 'pedestrian':
        # print(obj)
        # print('gt count: {}, pred: {}'.format(gt, num_centers))

        if i % 1000 == 0 and i != 0:
            print(i)
            # end_time = datetime.now()
            # time_diff = (end_time - start_time).total_seconds()
            # print('time: {}'.format(time_diff/100))
            # break

    file = 'result_output/result_center_partition2_overlap_0.3_dynamic_{}.txt'.format(thresh)
    with open(file, 'w') as f:
        json.dump(result_dict, f)
    return result_dict

def hm_partition(thresh=0.07, radius=1):
    objs = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    gt_path = 'gt_list.txt'
    if os.path.exists(gt_path):
            gt_list = json.load(open(gt_path))

    result_dict = {}
    start_time = datetime.now()
    for i in range(15050): #15050
        obj_idx = i % 10

        file_path = '/data1/hms_partition_2/{}.txt'.format(i)
        if os.path.exists(file_path):
            hms= json.load(open(file_path))
            # print(len(hms[0]))
            center_counts = torch.zeros(len(hms[0])) # batch size 4
            for hm in hms:
                hm= torch.tensor(hm, dtype=torch.float32)
                hm = sigmoid(hm)
                num_centers, padded_centers = extract_centers_dynamic(hm, thresh, radius=radius)

                if objs == 'motorcycle' or objs == 'bicycle':
                    num_centers, padded_centers = extract_centers_dynamic(hm, 0.12, radius=radius)

                center_counts += num_centers
                # print('hm shpe: ')
                # print(num_centers.shape)

            num_centers = center_counts.tolist()
            # print(num_centers)

            gt = gt_list[i]

            # if obj_idx == 0:
            #     print('gt {}, pred {}'.format(gt, num_centers))

            obj = objs[obj_idx]
            if obj in result_dict.keys():
                result_dict[obj]['gt_count'].extend(gt)
                result_dict[obj]['center_count'].extend(num_centers)
            else:
                result_dict[obj] = {'gt_count': gt, 'center_count': num_centers}
        else:
            print('not exist')

        # if obj == 'pedestrian':
        #     print('gt count: {}, pred: {}'.format(gt, num_centers))

        if i % 1000 == 0:
            print(i)


    file = 'result_output/result_center_partition_2_dynamic_{}_{}.txt'.format(thresh, radius)
    # file = 'result_output/result_center_partition_9_dynamic.txt'
    with open(file, 'w') as f:
        json.dump(result_dict, f)
    return result_dict

def test_time_partition(thresh, radius, path):
    print('thresh: {}, radius: {}'.format(thresh, radius))
    objs = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    gt_path = 'gt_list.txt'
    if os.path.exists(gt_path):
            gt_list = json.load(open(gt_path))

    result_dict = {}
    start_time = datetime.now()
    for i in range(15050): #15050
        obj_idx = i % 10
        # file_path = '/data/hms_partition_ep10/{}.txt'.format(i)
        # file_path = '/data/hms_partition_overlap9/{}.txt'.format(i)
        file_path = path + '{}.txt'.format(i)
        if os.path.exists(file_path):
            hms= json.load(open(file_path))
            # print(len(hms[0]))
            center_counts = torch.zeros(len(hms[0])) # batch size 4
            for hm in hms:
                hm= torch.tensor(hm, dtype=torch.float32)
                hm = sigmoid(hm)
                num_centers, padded_centers = extract_centers_dynamic(hm, thresh, radius=radius)

                if objs == 'motorcycle' or objs == 'bicycle':
                    num_centers, padded_centers = extract_centers_dynamic(hm, 0.12, radius=radius)

                center_counts += num_centers
                # print('hm shpe: ')
                # print(num_centers.shape)

            num_centers = center_counts.tolist()
            # print(num_centers)

            gt = gt_list[i]

            # if obj_idx == 0:
            #     print('gt {}, pred {}'.format(gt, num_centers))

            obj = objs[obj_idx]
            if obj in result_dict.keys():
                result_dict[obj]['gt_count'].extend(gt)
                result_dict[obj]['center_count'].extend(num_centers)
            else:
                result_dict[obj] = {'gt_count': gt, 'center_count': num_centers}
        else:
            print('not exist')

        # if obj == 'pedestrian':
        #     print('gt count: {}, pred: {}'.format(gt, num_centers))

        if i % 1000 == 0 and i != 0:
            print(i)
            end_time = datetime.now()

            time_diff = (end_time - start_time).total_seconds()
            print('time: {}'.format(time_diff/1000))

            break

def eval_combine(thresh, radius):
    object_combines = [['car', 'pedestrian'], ['car', 'barrier'], ['pedestrian', 'barrier'], ['car', 'pedestrian', 'barrier']]

    result_dict = load_eval_file('result_output/result_transfusion_0.02.txt')

    for combine in object_combines:
        frame_ids = len(result_dict['car']['center_count'])
        correct = 0

        for i in range(frame_ids):
            is_correct = True
            for item in combine:
                pred = result_dict[item]['center_count']
                gt = result_dict[item]['gt_count']

                diff = math.ceil(max(gt) * 0.1)

                if (gt[i] >= pred[i] - diff and gt[i] <= pred[i] + diff) == False:
                        is_correct = False

            if is_correct:
                correct += 1

        acc = correct/frame_ids
        print('category: {}, acc: {}'.format(combine, acc))

def eval_single_frame(model_name, frame_idx, objs):
    pred_list = []
    gt_list = []

    path = 'result_output/model_results/{}.txt'.format(model_name)
    result_dict = load_eval_file(path)
    # print(result_dict.keys())
    for key in objs:
        pred = result_dict[key]['center_count'][frame_idx]
        gt = result_dict[key]['gt_count'][frame_idx]

        pred_list.append(pred)
        gt_list.append(gt)
    return pred_list, gt_list

def getFeature():
    objs = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    model_list = ['partition_2_overlap_0.1', 'partition_2_overlap_0.2', 'partition_2_overlap_0.3','partition_2', 'partition_4_overlap_0.1', 'partition_4_overlap_0.2', 'partition_4_overlap_0.3', 'partition_4', 'partition_9_overlap_0.1','partition_9_overlap_0.2', 'partition_9_overlap_0.3','partition_9'] #11
    model_list = ['partition_2_overlap_0.1', 'partition_2_overlap_0.2', 'partition_2', 'partition_4_overlap_0.1', 'partition_4_overlap_0.2', 'partition_4', 'partition_9_overlap_0.1','partition_9_overlap_0.2', 'partition_9'] #12
    model_list = ['counternet', 'partition_2_overlap_0.1', 'partition_2_overlap_0.2', 'partition_2', 'partition_4_overlap_0.1', 'partition_4_overlap_0.2', 'partition_4', 'partition_9_overlap_0.1','partition_9_overlap_0.2', 'partition_9'] #12

    model_frame_dict = {string: [] for string in model_list}

    frame_idx = 0
    for i in range(14000): #15050
        # for each i, find 10 obj in each model
        file_path = '/data/feature_nus/' + '{}.txt'.format(i)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                batch_tensor = pickle.load(f)
                # print(batch_tensor.shape)

                for tensor in batch_tensor:

                    selected_model_idx = 0
                    min_diff = 1000
                    for m_idx, model in enumerate(model_list):
                        # get model
                        try:
                            preds, gts = eval_single_frame(model, frame_idx, objs) # expect to return count of all object and the corresponding gt in list
                        except IndexError:
                            break
                        # print(preds, gts)
                        diff = sum(abs(a - b) for a, b in zip(preds, gts))
                        # print('frame_id: {}, model: {}, diff: {}'.format(frame_idx, model, diff))
                        if diff < min_diff:
                            selected_model_idx = m_idx
                            min_diff = diff
                    model_frame_dict[model_list[selected_model_idx]].append(tensor.tolist())
                    frame_idx += 1

    for key in model_frame_dict.keys():
        print('key: {}, length: {}'.format(key, len(model_frame_dict[key])))
    # print(model_frame_dict)
    file = 'result_output/model_results/model_frame_dict13.txt'
    with open(file, 'w') as f:
        json.dump(model_frame_dict, f)

def getCenter():
    center_dict = {}
    file_path = 'result_output/model_results/model_frame_dict13.txt'
    if os.path.exists(file_path):
            model_frame_dict = json.load(open(file_path))
            for key in model_frame_dict.keys():
                data_array = np.array(model_frame_dict[key])
                avg_pool = np.mean(data_array, axis=0)

                distance_sum = 0
                for array in data_array:
                    distance = np.linalg.norm(array - avg_pool)
                    distance_sum += distance
                avg_distance = distance_sum/len(data_array) # sigma square
                print(avg_distance)

                # adjustment chernoff bound
                epsilon = 0.15
                # print('key: {}, avg_dist: {}'.format(key, avg_distance))
                power = -(len(data_array) * epsilon*epsilon) / (2 * avg_distance)
                adjustment = math.exp(power)
                print('key: {}, adjustment: {}'.format(key, adjustment))
                # model_frame_dict[key] = avg_pool.tolist()

                if key not in center_dict.keys():
                    center_dict[key] = {'center': avg_pool, 'adjustment': adjustment}
    return center_dict
# def combineResults():

def evalModelSelection():
    center_dict = getCenter()
    output_dict = {}
    objs = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    model_list = ['counternet', 'partition_2_overlap_0.1', 'partition_2_overlap_0.2', 'partition_2', 'partition_4_overlap_0.1', 'partition_4_overlap_0.2', 'partition_4', 'partition_9_overlap_0.1','partition_9_overlap_0.2', 'partition_9'] #13

    frame_idx = 0
    model_count = {}
    for i in range(15050): #15050
        # for each i, find 10 obj in each model
        file_path = '/data/feature_nus/' + '{}.txt'.format(i)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                batch_tensor = pickle.load(f)
                # print(batch_tensor.shape)
                for tensor in batch_tensor:
                    selected_model = model_list[0]
                    distance_min = 1000
                    for m_idx, model in enumerate(model_list):
                        center = center_dict[model]['center'] #'adjustment'
                        adjustment = center_dict[model]['adjustment']
                        distance = np.linalg.norm(center - tensor.tolist())
                        # distance += adjustment * distance
                        # print('model: {}, adjust: {}, distance: {}'.format(model, adjustment, distance))
                        if distance < distance_min:
                            distance_min = distance
                            selected_model = model

                    if selected_model not in model_count.keys():
                        model_count[selected_model] = 1
                    else:
                        model_count[selected_model] += 1


                    try:
                        preds, gts = eval_single_frame(selected_model, frame_idx, objs) # expect to return count of all object and the corresponding gt in list
                    except IndexError:
                        break

                    for idx, pred in enumerate(preds):
                        obj = objs[idx]
                        if obj not in output_dict.keys():
                            output_dict[obj] = {'gt_count': [gts[idx]], 'center_count':[pred]}
                        else:
                            output_dict[obj]['gt_count'].append(gts[idx])
                            output_dict[obj]['center_count'].append(pred)
                    frame_idx += 1
    # print(frame_idx)
    # print(len(output_dict['pedestrian']['gt_count']))
    print(model_count)
    file = 'result_output/model_results/result_model_selection13.txt'
    with open(file, 'w') as f:
        json.dump(output_dict, f)

def eval(thresh, radius, ep=0.1):

    model_list = ['partition_2_overlap_0.2', 'partition_2', 'partition_4_overlap_0.1', 'partition_4_overlap_0.2', 'partition_4_overlap_0.3', 'partition_4', 'partition_9_overlap_0.2', 'partition_9']
    result_dict = load_eval_file('result_output/model_results/{}.txt'.format(model_list[4]))

    obj_list = []
    acc_list = []

    print(result_dict.keys())
    for key in result_dict.keys():
        pred = result_dict[key]['center_count']
        gt = result_dict[key]['gt_count']

        # print('pred:  {}, gt: {}'.format(pred, gt))
        # acc = accuracy_score(gt, pred)
        # print('category: {}, acc: {}'.format(key, acc))
        correct = 0
        print(key)

        diff = math.ceil(max(gt) * 0.1)
        # print('diff {}'.format(diff))
        for i in range(len(gt)):
            if gt[i] >= pred[i] - diff and gt[i] <= pred[i] + diff:
                correct += 1

        acc = correct/len(pred)
        print('category: {}, acc: {}'.format(key, acc))

        obj_list.append(key)
        acc_list.append(acc)

    return obj_list,acc_list

def eval_binary(thresh, radius):
    model_list = ['partition_2_overlap_0.2', 'partition_2', 'partition_4_overlap_0.1', 'partition_4_overlap_0.2', 'partition_4_overlap_0.3', 'partition_4', 'partition_9_overlap_0.2', 'partition_9']
    result_dict = load_eval_file('result_output/model_results/{}.txt'.format(model_list[4]))
    obj_list = []
    acc_list = []

    print(result_dict.keys())
    for key in result_dict.keys():
        pred = result_dict[key]['center_count']
        gt = result_dict[key]['gt_count']

        correct = 0
        total = 0
        print(key)

        diff = 0 #math.ceil(max(gt) * 0.05)

        for i in range(len(gt)):
            if gt[i] != 0 and pred[i] != 0:
                correct += 1
            if gt[i] == 0 and pred[i] == 0:
                correct += 1

        print('total {}'.format(total))
        acc = correct/len(pred)
        print('category: {}, acc: {}'.format(key, acc))

        obj_list.append(key)
        acc_list.append(acc)

    return obj_list,acc_list

def eval_count(thresh, radius):
    model_list = ['partition_2_overlap_0.2', 'partition_2', 'partition_4_overlap_0.1', 'partition_4_overlap_0.2', 'partition_4_overlap_0.3', 'partition_4', 'partition_9_overlap_0.2', 'partition_9']
    result_dict = load_eval_file('result_output/model_results/{}.txt'.format(model_list[4]))
    obj_list = []
    acc_list = []

    print(result_dict.keys())
    for key in result_dict.keys():
        pred = result_dict[key]['center_count']
        gt = result_dict[key]['gt_count']

        correct = 0
        total = 0
        print(key)

        diff = math.ceil(max(gt) * 0.05)

        total_acc = 0
        for j in range(1000):
            correct = 0
            total = 0
            random_number = random.randint(0, len(gt) - 1 - 250)
            for i in range(random_number, random_number + 250 +1):
                if gt[i] == 5:
                    total += 1
                    # if gt[i] >= pred[i] - diff and gt[i] <= pred[i] + diff:
                    if gt[i] >= pred[i] - diff and gt[i] <= pred[i] + diff:
                        correct += 1
            if total == correct:
                total_acc += 1
            else:
                total_acc += correct/total


        acc = total_acc/1000 #correct/len(pred)
        print('category: {}, acc: {}'.format(key, acc))

        obj_list.append(key)
        acc_list.append(acc)

    return obj_list,acc_list

def eval_agg(thresh, radius):
    model_list = ['partition_2_overlap_0.2', 'partition_2', 'partition_4_overlap_0.1', 'partition_4_overlap_0.2', 'partition_4_overlap_0.3', 'partition_4', 'partition_9_overlap_0.2', 'partition_9']
    result_dict = load_eval_file('result_output/model_results/{}.txt'.format(model_list[4]))
    obj_list = []
    acc_list = []

    print(result_dict.keys())
    for key in result_dict.keys():
        pred = result_dict[key]['center_count']
        gt = result_dict[key]['gt_count']

        correct = 0
        total = 0
        print(key)

        diff = math.ceil(max(gt) * 0.05)

        total_diff_avg = 0
        for j in range(1000):
            gt_sum = 0
            pred_sum = 0
            random_number = random.randint(0, len(gt) - 1 - 250)
            for i in range(random_number, random_number + 250 +1):
                gt_sum += gt[i]
                pred_sum += pred[i]

            total_diff_avg += abs(gt_sum - pred_sum)/250


        acc = total_diff_avg/1000
        print('category: {}, acc: {}'.format(key, acc))

        obj_list.append(key)
        acc_list.append(acc)

    return obj_list,acc_list


def getCategoricalAcc(ground_truth, predictions):
    # Get unique categories
    categories = np.unique(ground_truth + predictions)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(ground_truth, predictions, labels=categories)

    # Calculate accuracy for each category
    category_accuracies = {}
    for i, category in enumerate(categories):
        true_positives = conf_matrix[i, i]
        total_samples = sum(conf_matrix[i, :])
        accuracy = true_positives / total_samples if total_samples != 0 else 0
        category_accuracies[category] = accuracy

    return category_accuracies


eval(0.58, 2)
# eval_binary(0.58, 2)
# eval_count(0.58, 2)
# eval_agg(0.58, 2)
# eval_combine(0.58, 2)

# result_dict = getData2()
# write_to_file(result_dict, 'thresh_0.2')
# # hm()

# getFeature()
# getCenter()
# evalModelSelection()


# hm_partition()
# hm_partition_overlap()
# hm()
