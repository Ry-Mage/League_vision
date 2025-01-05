import cv2
import numpy as np
import os
import time
import pandas as pd
import os
import json
import math
from ultralytics import YOLO
import torch
from natsort import natsorted
import keras
#from keras import layers
from keras import Model
from keras.api.layers import Conv2D, MaxPool2D, Reshape, Input, ReLU
from keras.api.activations import sigmoid, softmax, relu
from keras.api.losses import SparseCategoricalCrossentropy
import torch.nn.functional as F

cuda = torch.device('cuda:0')
# League computer vision

# Probably regression based? Could output coordinates on league map (?) Could potentially also use classification, but would need classes for many many areas of the map.
#### PERHAPS SEMANTIC SEGMENTATION???



# Basic CNN Example:
# Convolutional layer 1 (ex 32 neurons, ReLU activation)
# Pooling Layer 1
# Convolutional Layer 2 (ex 64 neurons, ReLU activation)
# Pooling layer 2
##### Above here is somewhat diverse/able to be applied to many different problems. Used to identify/extract features, below this is specifically for classification using those features
# Flatten layers (keras.layers.Flatten())
# Dense layer 1 (ex 1024 neurons, ReLU activation)
# Dense layer 2 (ex 10 neurons, softmax activation) <--- Output layer, with 10 possible classification outputs

# Input Data:
# 1. Could just use the DeepLeague dataset
# 2. Could create a dataset of images where each champion icon is overlayed on the minimap at many possible positions, with the same number of images per champ


# Issues with DeepLeague?
# # Model is trained on where champs have existed previously on the map of an LCS game, this means it wont generalize well. Not only is the occurence of champs in the dataset going to be imbalanced but also where they appear

# Steps to implement a YOLO-like champ detector for league minimap
# 1. Create dataset. Overlay league champ icons (with roughly equal proportions for each champ) onto the minimap in different positions. Include bounding box information (?) and which champ it is as part of this data. These boundary boxes should have confidence score and class probabilities of 1 as "ground truth"
# 2. Perhaps implement a yolo-like loss function
# 3. Implement model/network architecture. Multiple convolutional layers and maxpool, separated by 1x1 "downscaling" convolutional layers and 3x3 conv. layers, concluding with 2 fully connected layers

#### Can perhaps use an Ultralytics YOLO model, either for final implementation or to later flesh out


# Takes lists of coordinates (should be between 433x433) and lists of champs, then overlays them onto a 1440p sized league minimap in one image. 
t = 0
homePath = 'C:/Code and Scoo/Kaggle and Data Sci/League Vision'
def overlay_champs(xs, ys, icons, res):
    timeOverS = time.perf_counter()
    background = cv2.imread(f"{homePath}/{res}_Minimap.png")
    for x, y, icon in zip(xs, ys, icons):
        overlay = cv2.imread(f"{homePath}/Champ Icon Circles/{icon}", cv2.IMREAD_UNCHANGED)
        if res == 1440:
            overlay = cv2.resize(overlay, dsize = (36, 36), interpolation = cv2.INTER_AREA)
        elif res == 1080:
            overlay = cv2.resize(overlay, dsize = (27, 27), interpolation = cv2.INTER_AREA)
        #Separates the "alpha" and color channels, and converts alpha to 0-1 instead of 0-255. Alpha being the transparency(?)
        alpha_channel = overlay[:, :, 3] / 255
        overlay_colors = overlay[:, :, :3]

        #Creates a mask for the alpha channel in the same shape as the colors of the image to overlay, duplicating the alpha channel for each color channel
        alpha_mask = alpha_channel[:, :, np.newaxis]

        # Takes a subsection of the background image to match dimensions of the overlaid image
        h, w = overlay.shape[:2]
        bh, bw = background.shape[:2]
        background_subsection = background[y:y + h, x:x + w]

        # Combine the background with the overlaid image with alpha channel as a weighting and updates background image at the overlay location
        composite = background_subsection * (1 - alpha_mask) + overlay_colors * alpha_mask
        background[y:y + h, x:x + w] = composite
    global t
    cv2.imwrite(f"{homePath}/Created/combined{t}_{res}.png", background)
    timeOverE = time.perf_counter()
    print(f'Composite {t} took {timeOverE - timeOverS} seconds')
    t += 1
    #cv2.imshow('Overlaid', background)
    #cv2.waitKey(0)


# Creates a whole dataset of minimap images with different champions at various positions. Should represent roughly equal proportions of each champion. Also creates an accompanying COCO formatted JSON file with annotations of classes (champs) and bounding boxes for each image
# Also creates an Ultralytics 'YOLO' formatted set of annotations
def create_image_set(setSize, champsPerSet, res):
    champs = os.listdir(f'{homePath}/Champ Icon Circles')
    rng = np.random.default_rng()
    allSelections = []
    allXCoords = []
    allYCoords = []
    classIds = dict([x, str(name[:name.index('_')])] for name, x in zip(os.listdir(f'{homePath}/Champ Icon Circles'), range(0, len(os.listdir(f'{homePath}/Champ Icon Circles')))))
    classIds = pd.DataFrame.from_dict(data = classIds, orient = 'index', columns = ['name'])
    classIds['id'] = classIds.index
    classIds = classIds[['id', 'name']]
    annotations = pd.DataFrame(columns = ['id', 'image_id', 'category_id', 'class_prob', 'confidence', 'bbox'])
    images = pd.DataFrame(columns = ['id', 'file_name', 'width', 'height', 'license'])
    yoloAnnots = pd.DataFrame(columns = ['class', 'xCentNorm', 'yCentNorm', 'wNorm', 'hNorm', 'imageName'])
    if res == 1440:
        for image in range(0, setSize):
            champSelection = rng.choice(a = champs, size = champsPerSet, replace = False)
            xCoords = rng.integers(20, 390, champsPerSet)
            yCoords = rng.integers(20, 390, champsPerSet)
            overlay_champs(xCoords, yCoords, champSelection, res)
            champSelection = [str(champ[:champ.index('_')]) for champ in champSelection]
            xCoords = [int(x) for x in xCoords]
            yCoords = [int(y) for y in yCoords]
            allSelections = allSelections + champSelection
            allXCoords = allXCoords + xCoords
            allYCoords = allYCoords + yCoords
        for tChamp in range(0, setSize*champsPerSet):
            #annotations.loc[tChamp] = [tChamp, 'combined' + str(math.floor(tChamp / 10)) + '_'+ str(res), classIds[classIds['name'] == allSelections[tChamp]].id.iloc[0], 1, 1, [allXCoords[tChamp], allYCoords[tChamp], 40, 40]]
            annotations.loc[tChamp] = [tChamp, math.floor(tChamp / 10), classIds[classIds['name'] == allSelections[tChamp]].id.iloc[0], 1, 1, [allXCoords[tChamp], allYCoords[tChamp], 40, 40]]
        for tImage in range(0, setSize):
            images.loc[tImage] = [tImage, 'combined' + str(tImage) + '_' + str(res) + '.png', 457, 460, 0]
        for anno in range(0, setSize * champsPerSet):
            xMin = allXCoords[anno]
            yMin = allYCoords[anno]
            xMax = allXCoords[anno] + 40
            yMax = allYCoords[anno] + 40
            xMin = max(0, xMin)
            yMin = max(0, yMin)
            xMax = min(457, xMax)
            yMax = min(460, yMax)
            actualW = xMax - xMin
            actualH = yMax - yMin
            actualXCent = xMin + actualW / 2
            actualYCent = yMin + actualH / 2
            xCentNorm = actualXCent / 457
            yCentNorm = actualYCent / 460
            actualWNorm = actualW / 457
            actualHNorm = actualH / 460
            xCentNorm = min(max(xCentNorm, 0), 1)
            yCentNorm = min(max(yCentNorm, 0), 1)
            actualWNorm = min(max(actualWNorm, 0), 1)
            actualHNorm = min(max(actualHNorm, 0), 1)
            yoloAnnots.loc[anno] = [classIds[classIds['name'] == allSelections[anno]].id.iloc[0], xCentNorm, yCentNorm, actualWNorm, actualHNorm, 'combined' + str(math.floor(anno / 10)) + '_' + str(res)]


    if res == 1080:
        for image in range(0, setSize):
            champSelection = rng.choice(a = champs, size = champsPerSet, replace = False)
            xCoords = rng.integers(16, 276, champsPerSet)
            yCoords = rng.integers(16, 276, champsPerSet)
            overlay_champs(xCoords, yCoords, champSelection, res)
            champSelection = [str(champ[:champ.index('_')]) for champ in champSelection]
            xCoords = [int(x) for x in xCoords]
            yCoords = [int(y) for y in yCoords]
            allSelections += champSelection
            allXCoords += xCoords
            allYCoords += yCoords
        for tChamp in range(0, setSize*champsPerSet):
            annotations.loc[tChamp] = [tChamp, math.floor(tChamp / 10), classIds[classIds['name'] == allSelections[tChamp]].id.iloc[0], 1, 1, [allXCoords[tChamp], allYCoords[tChamp], 30, 30]]
        for tImage in range(0, setSize):
            images.loc[tImage] = [tImage, 'combined' + str(tImage) + '_' + str(res) + '.png', 342, 342, 0]
        for anno in range(0, setSize * champsPerSet):
            yoloAnnots.loc[anno] = [classIds[classIds['name'] == allSelections[anno]].id.iloc[0], (allXCoords[anno] + 15) / 342, (allYCoords[anno] + 15) / 342, 30 / 342, 30 / 342, 'combined' + str(math.floor(anno / 10)) + '_'+ str(res)]


    classIdsJson = classIds.to_json(orient = 'records')
    annotationsJson = annotations.to_json(orient = 'records')
    imagesJson = images.to_json(orient = 'records')
    combinedJson = f'"images" : {imagesJson},\n"annotations" : {annotationsJson},\n"categories" : {classIdsJson}'
    cocoJsonS = '{"info" : {"description" : "custom league of legends minimap dataset", "url": "https://google.com","version" : 0.1,"year" : 2024, "contributor": "Ryan", "date_created": "11/29/2024"},"licenses" : [{"url" : "https://www.gnu.org/licenses/agpl-3.0.en.html","id" : 0,"name" : "GNU Affero General Public License"}],'
  
    cocoJsonS = cocoJsonS + combinedJson + '}'
    annotationsJson = "annotations : " + annotationsJson
    cocoJson = json.loads(cocoJsonS)
    #print(cocoJson)

    # with open ('coco_annotations.json', 'w') as json_file:
    #     json.dump(cocoJson, json_file)
    # convert_coco(labels_dir = "C:/Code and Scoo/Kaggle and Data Sci/coco_annotations.json")

    for imag in yoloAnnots.groupby('imageName'):
        with open(imag[0]+ '.txt', 'w') as file:
             file.write(imag[1][['class', 'xCentNorm', 'yCentNorm', 'wNorm', 'hNorm']].to_string(header = False, index = False))
             file.close()

#create_image_set(200, 10, 1440)

#overlay_champs((150, 200), (150, 200), ("aatrox", "ahri"))
#overlay_champs((150, 200), (150, 200), ("aatrox", "shen"))


def ultralytics_yolo():
    if __name__ == '__main__':
        model = YOLO("C:/Code and Scoo/Kaggle and Data Sci/Models/Ultralytics/yolo11n.pt")
        hist = model.train(data = "C:/Code and Scoo/Kaggle and Data Sci/League Vision/Test/data.yaml", lr0 = 0.1, lrf = 0.8, box = 1.5, cls = 2, plots = True, dfl = 0.4, optimizer = 'AdamW')
        metrics = model.val()
        metrics.box.map
    return




# Read all image + annotation files, split into test and train sets. Assumes images and annotations are already separated into train/test in folders. Ie /images/train, /labels/val, etc
def read_dataset(homePath):
    imagePathsTrain = natsorted([homePath + '/images/train/' + it for it in os.listdir(homePath+'/images/train')])
    imagePathsVal = natsorted([homePath + '/images/val/' + iv for iv in os.listdir(homePath+'/images/train')])
    imagesTrain = [cv2.imread(it) for it in imagePathsTrain]
    imagesVal = [cv2.imread(iv) for iv in imagePathsVal]
    annotsPathsTrain = natsorted([homePath + '/labels/train/' + at for at in os.listdir(homePath + '/labels/train')])
    annotsPathsVal = natsorted([homePath + '/labels/val/' + av for av in os.listdir(homePath + '/labels/val')])
    annotsTrain = []
    annotsVal = []
    for annot in enumerate(annotsPathsTrain):
        temp = []
        with open(annot[1], 'r') as file:
            for line in file:
                temp.append(line.strip().split())
            annotsTrain.append(temp)
    annotsTrain = [[[int(float(value)) if int(float(value)) == float(value) else float(value) for value in sublist] for sublist in subblist] for subblist in annotsTrain]
    for annot in enumerate(annotsPathsVal):
        temp = []
        with open(annot[1], 'r') as file:
            for line in file:
                temp.append(line.strip().split())
            annotsVal.append(temp)
    annotsVal = [[[int(float(value)) if int(float(value)) == float(value) else float(value) for value in sublist] for sublist in subblist] for subblist in annotsVal]
    
    return np.array(imagesTrain), np.array(annotsTrain), np.array(imagesVal), np.array(annotsVal)
    
xTrain, yTrain, xTest, yTest = read_dataset('C:/Code and Scoo/Kaggle and Data Sci/League Vision/Test')

print(f'Training images length: {len(xTrain)}, training images shape {xTrain[0].shape}, Training labels length: {len(yTrain)}, Test images length: {len(xTest)}, training images shape {xTest[0].shape}, Training labels length: {len(yTest)}')

def generate_anchors(stride = 2, feature_height = 29, feature_width = 29, anchor_w = 40, anchor_h = 40, imageW = 457, imageH = 460):
    anchors = []
    anchor_w = anchor_w / imageW
    anchor_h = anchor_h / imageH
    for h in range(feature_height):
        for w in range(feature_width):
            cx = ((w/29)+0.0172) 
            cy = ((h/29)+0.0172) 
            anchors.append([cx, cy, anchor_w, anchor_h])
    return torch.from_numpy(np.array(anchors)).to(cuda)




def create_detection_head(featureMapLayer, num_anchors = 841, num_classes = 173):
    x = Conv2D(256, 3, activation = 'relu', padding = 'same')(featureMapLayer)

    # Champ Classification 
    class_logits = Conv2D(num_classes, 1)(x)
    class_logits = Reshape((-1, num_classes))(class_logits)

    #Bbox offset regression
    bbox_offset = Conv2D(4, 1)(x) 
    bbox_offset = Reshape((-1, 4))((bbox_offset))
    return class_logits, bbox_offset

def create_model(inputSize):
    inputLayer = Input(shape = inputSize)
    # Downsample 460 -> 230
    firstDownsample = Conv2D(64, kernel_size = 3, strides = 2, padding = 'same')(inputLayer)
    firstDownsample = ReLU()(firstDownsample)
    # Downsample 230 -> 115
    secondDownsample = Conv2D(128, kernel_size = 3, strides = 2, padding = 'same')(firstDownsample)
    secondDownsample = ReLU()(secondDownsample)
    # Downsample 115 -> 57.5
    thirdDownsample = Conv2D(256, kernel_size = 3, strides = 2, padding = 'same')(secondDownsample)
    thirdDownsample = ReLU()(thirdDownsample)
    # Downsample 57.5 -> 28.75
    fourthDownsample = Conv2D(512, kernel_size = 3, strides = 2, padding = 'same')(thirdDownsample)
    featureLayer = ReLU()(fourthDownsample)
    class_out, regress_out = create_detection_head(featureLayer)
    return Model(inputLayer, [class_out, regress_out])
cModel = create_model((460, 457, 3))



def centerwh_to_xyxy_vectorized(boxes):
    x_c = boxes[:, 0]
    y_c = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    
    return torch.stack([x1, y1, x2, y2], dim = 1)

def box_iou_vectorized(gtBoxes, anchorBoxes):
    N = anchorBoxes.shape[0]
    M = gtBoxes.shape[0]

    x1_g = gtBoxes[:, 0].unsqueeze(0).expand(N, M)
    y1_g = gtBoxes[:, 1].unsqueeze(0).expand(N, M)
    x2_g = gtBoxes[:, 2].unsqueeze(0).expand(N, M)
    y2_g = gtBoxes[:, 3].unsqueeze(0).expand(N, M)

    x1_a = anchorBoxes[:, 0].unsqueeze(1).expand(N, M)
    y1_a = anchorBoxes[:, 1].unsqueeze(1).expand(N, M)
    x2_a = anchorBoxes[:, 2].unsqueeze(1).expand(N, M)
    y2_a = anchorBoxes[:, 3].unsqueeze(1).expand(N, M)

    inter_x1 = torch.max(x1_g, x1_a)
    inter_y1 = torch.max(y1_g, y1_a)
    inter_x2 = torch.max(x2_g, x2_a)
    inter_y2 = torch.max(y2_g, y2_a)

    inter_w = (inter_x2 - inter_x1).clamp(min = 0)
    inter_h = (inter_y2 - inter_y1).clamp(min = 0)
    inter_area = inter_w * inter_h

    area1 = (x2_g - x1_g).clamp(min = 0) * (y2_g - y1_g).clamp(min = 0)
    area2 = (x2_a - x1_a).clamp(min = 0) * (y2_a - y1_a).clamp(min = 0)
    union_area = area1 + area2 - inter_area

    IOUs = inter_area / union_area

    return IOUs

def match_gtBbox_anchors(gtBoxes, anchorBoxes, iou_thresh = 0.5, background_class = 172):
    gt_clss = gtBoxes[:, 0].long()
    gtBoxes = gtBoxes[:, 1:]

    gtBoxes = centerwh_to_xyxy_vectorized(gtBoxes)
    anchorBoxes = centerwh_to_xyxy_vectorized(anchorBoxes)

    N = anchorBoxes.shape[0]
    M = gtBoxes.shape[0]

    ious = box_iou_vectorized(gtBoxes, anchorBoxes)

    iou_vals, gt_indices = ious.max(dim = 1)

    fg_mask = iou_vals >= iou_thresh

    cls_targets = torch.full((N,), background_class, dtype=torch.long, device = cuda)
    cls_targets[fg_mask] = gt_clss[gt_indices[fg_mask]]

    offsets = torch.zeros((N, 4), dtype = torch.double, device = cuda)
    
    fg_anchors = anchorBoxes[fg_mask]
    gt_fg = gtBoxes[gt_indices[fg_mask]]
    offsets_fg = gt_fg - fg_anchors
    offsets[fg_mask] = offsets_fg


    return cls_targets, offsets, fg_mask




def classification_loss(y_true_cls, y_pred_cls):

    B, A, C = y_pred_cls.shape
    pred_flat = y_pred_cls.view(B*A, C)
    labels_flat = y_true_cls.view(B*A)
    
    lossT = F.cross_entropy(pred_flat, labels_flat)

    return torch.mean(lossT)

def bbox_regression_loss(y_true_regr, y_pred_regr, anchorMask):
    diff = y_pred_regr - y_true_regr
    abs_diff = torch.abs(diff)
    smooth_l1 = torch.where(abs_diff < 1, 0.5 * abs_diff ** 2, abs_diff - 0.5)
    smooth_l1 = smooth_l1 * torch.unsqueeze(anchorMask, axis = -1)
    return torch.mean(smooth_l1)

def total_loss(y_true_cls, y_pred_cls, y_true_regr, y_pred_regr, anchorMask):
    cls_loss = classification_loss(y_true_cls, y_pred_cls)
    regr_loss = bbox_regression_loss(y_true_regr, y_pred_regr, anchorMask)
    return cls_loss + regr_loss


def train_model(epochs):
    batch_size = 32
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(xTrain), torch.from_numpy(yTrain)
    ) 
    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(xTest), torch.from_numpy(yTest)
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size = batch_size, shuffle = True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size = batch_size, shuffle = False 
    )
    model = cModel
    cModel.to()
    optimizer = keras.optimizers.Adam(learning_rate = 1e-3)
    anchors = generate_anchors()
    for epoch in range(epochs):
        print(f'\n Start of epoch: {epoch}')
        for step, (batch_imgs, batch_boxes) in enumerate(train_dataloader):
            
            all_cls_targets = []
            all_box_targets = []
            all_fg_masks = []
            batch_boxes = batch_boxes.type(torch.LongTensor).to(cuda)
            
            cls_pred, regr_pred = cModel(batch_imgs)
            for bID in range(batch_imgs.size(0)):
                gtBboxes = batch_boxes[bID]
                y_true_cls, y_true_regr, anchorForegroundMask = match_gtBbox_anchors(gtBboxes, anchors)
                all_cls_targets.append(y_true_cls)
                all_box_targets.append(y_true_regr)
                all_fg_masks.append(anchorForegroundMask)

            fg_masks_t = torch.tensor(torch.stack(all_fg_masks), dtype = bool).to(cuda)
            cls_targets_t = torch.tensor(torch.stack(all_cls_targets)).type(torch.LongTensor).to(cuda)
            box_targets_t = torch.tensor(torch.stack(all_box_targets)).type(torch.FloatTensor).to(cuda)


            loss = total_loss(cls_targets_t, cls_pred, box_targets_t, regr_pred, fg_masks_t)
            if model.losses:
                loss = loss + torch.sum(*model.losses)
            model.zero_grad()
            trainable_weights = [w for w in model.trainable_weights]
            loss.backward()
            gradients = [v.value.grad for v in trainable_weights]

            with torch.no_grad():
                optimizer.apply(gradients, trainable_weights)
            
            #print(f'Step: {step}')
            if step % 10 == 0:
                print(f'Training loss (for 1 batch) at step {step}: {loss.cpu().detach().numpy():.4f}')
                print(f'Seen so far {(step+1)*batch_size} samples')

        # for batch_imgs_val, batch_boxes_val in val_dataloader:
        #     val_cls, val_regr = model(batch_imgs_val, training = False)
        #     y_true_val_cls, y_true_val_regr, anchorForegroundMask = bbox_anchor_match(batch_boxes_val, anchors)
        #     val_loss = total_loss(y_true_val_cls, val_cls, y_true_val_regr, val_regr, anchorForegroundMask)
        #     print(f'Validation loss: {val_loss}')

train_model(500)


#TODO
def visualize_predictions(image, boxes, labels, class_names):
    vis_img = image.copy()
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].int()
        
        cls_id = labels[i].item()

    return vis_img



#TODO
# Inference (Decoding and non-maximal suppression): Forward pass, decode bbox coordinates from anchors + predicted offsets, select top n-number of class predictions by confidence or apply a threshold. Use NMS to remove duplicate identifications. Keep bboxes with high confidence or until we hit 10 boxes
# 