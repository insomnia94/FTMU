from data import COCODetection, MEANS, COLORS, COCO_CLASSES
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from pathlib import Path
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import cv2
import copy

from torchvision import transforms
import torchvision.models as models

import torchvision.models as models

import os
import shutil
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import copy

from parameters import *

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
  parser = argparse.ArgumentParser(
    description='YOLACT COCO Evaluation')
  parser.add_argument('--trained_model',
                      default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                      help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
  parser.add_argument('--top_k', default=5, type=int,
                      help='Further restrict the number of predictions to parse')
  parser.add_argument('--cuda', default=True, type=str2bool,
                      help='Use cuda to evaulate model')
  parser.add_argument('--cross_class_nms', default=True, type=str2bool,
                      help='Whether to use cross-class nms (faster) or do nms per class')
  parser.add_argument('--fast_nms', default=True, type=str2bool,
                      help='Whether to use a faster, but not entirely correct version of NMS.')
  parser.add_argument('--display_masks', default=True, type=str2bool,
                      help='Whether or not to display masks over bounding boxes')
  parser.add_argument('--display_bboxes', default=True, type=str2bool,
                      help='Whether or not to display bboxes around masks')
  parser.add_argument('--display_text', default=True, type=str2bool,
                      help='Whether or not to display text (class [score])')
  parser.add_argument('--display_scores', default=True, type=str2bool,
                      help='Whether or not to display scores in addition to classes')
  parser.add_argument('--display', dest='display', action='store_true',
                      help='Display qualitative results instead of quantitative ones.')
  parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                      help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
  parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                      help='In quantitative mode, the file to save detections before calculating mAP.')
  parser.add_argument('--resume', dest='resume', action='store_true',
                      help='If display not set, this resumes mAP calculations from the ap_data_file.')
  parser.add_argument('--max_images', default=-1, type=int,
                      help='The maximum number of images from the dataset to consider. Use -1 for all.')
  parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                      help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
  parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                      help='The output file for coco bbox results if --coco_results is set.')
  parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                      help='The output file for coco mask results if --coco_results is set.')
  parser.add_argument('--config', default=None,
                      help='The config object to use.')
  parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                      help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
  parser.add_argument('--web_det_path', default='web/dets/', type=str,
                      help='If output_web_json is set, this is the path to dump detections into.')
  parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                      help='Do not output the status bar. This is useful for when piping to a file.')
  parser.add_argument('--display_lincomb', default=False, type=str2bool,
                      help='If the config uses lincomb masks, output a visualization of how those masks are created.')
  parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                      help='Equivalent to running display mode but without displaying an image.')
  parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                      help='Do not sort images by hashed image ID.')
  parser.add_argument('--seed', default=None, type=int,
                      help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
  parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                      help='Outputs stuff for scripts/compute_mask.py.')
  parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                      help='Do not crop output masks with the predicted bounding box.')
  parser.add_argument('--image', default=None, type=str,
                      help='A path to an image to use for display.')
  parser.add_argument('--images', default=None, type=str,
                      help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
  parser.add_argument('--video', default=None, type=str,
                      help='A path to a video to evaluate on.')
  parser.add_argument('--video_multiframe', default=1, type=int,
                      help='The number of frames to evaluate in parallel to make videos play at higher fps.')
  parser.add_argument('--score_threshold', default=0, type=float,
                      help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
  parser.add_argument('--dataset', default=None, type=str,
                      help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
  parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                      help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')

  parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                      benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False)

  global args
  args = parser.parse_args(argv)

  if args.output_web_json:
    args.output_coco_json = True

  if args.seed is not None:
    random.seed(args.seed)

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = [] # Call prep_coco_cats to fill this
coco_cats_inv = {}

# customized help function

# this function is used to generate the bounding box of all smaller bounding boxes
def generate_ROI(box_list, original_w, original_h, ratio):

  top_x_list = []
  top_y_list = []
  bottom_x_list = []
  bottom_y_list = []

  for i in range(len(box_list)):
    top_x_list.append(box_list[i][0])  # x1
    top_y_list.append(box_list[i][1])  # y1
    bottom_x_list.append(box_list[i][2])  # x2
    bottom_y_list.append(box_list[i][3])  # y2

  # the bounding box of all bounding boxes of ground-truth
  top_x = min(top_x_list)
  top_y = min(top_y_list)
  bottom_x = max(bottom_x_list)
  bottom_y = max(bottom_y_list)

  w = bottom_x - top_x
  h = bottom_y - top_y

  top_x = int(top_x - ratio * w)
  top_y = int(top_y - ratio * h)
  bottom_x = int(bottom_x + ratio * w)
  bottom_y = int(bottom_y + ratio * w)

  if top_x < 0:
    top_x = 0
  if top_y < 0:
    top_y = 0
  if bottom_x > original_w:
    bottom_x = original_w
  if bottom_y > original_h:
    bottom_y = original_h

  return top_x, top_y, bottom_x, bottom_y

def mask_divided(mask):
  mask_list = []
  mask_shape = mask.shape
  mask_h = mask_shape[0]
  mask_w = mask_shape[1]

  # all possible values referring to different objects
  possible_values = np.unique(mask)

  # remove the first value cos it is 0
  possible_values = possible_values[1:]

  # the number of all possible values
  num_values = len(possible_values)

  for i in range(num_values):
    # True or False matrix
    bool_matrix = mask == possible_values[i]

    # 0 or 255 matrix
    binary_matrix = np.zeros((mask_h, mask_w)) + np.full((mask_h, mask_w), 255) * bool_matrix
    binary_matrix = binary_matrix.astype(np.uint8)

    mask_list.append(binary_matrix)

  return possible_values, mask_list

def mask2box(mask):
  # a list to store the x,y,w,h for each bounding box
  box_list = []

  mask_shape = mask.shape
  mask_h = mask_shape[0]
  mask_w = mask_shape[1]

  # all possible values referring to different objects
  possible_values = np.unique(mask)

  # remove the first value cos it is 0
  possible_values = possible_values[1:]

  # the number of all possible values
  num_values = len(possible_values)

  for i in range(num_values):
    # True or False matrix
    bool_matrix = mask == possible_values[i]

    # 0 or 255 matrix
    binary_matrix = np.zeros((mask_h, mask_w)) + np.full((mask_h, mask_w), 255) * bool_matrix
    binary_matrix = binary_matrix.astype(np.uint8)

    _, contours = cv2.findContours(binary_matrix, 3, 2)

    contour_box_list = []

    for contour_id in range(len(contours)):
      x, y, w, h = cv2.boundingRect(contours[contour_id])
      contour_box_list.append([x, y, x+w, y+h])

    top_x, top_y, bottom_x, bottom_y = generate_ROI(contour_box_list, mask_w, mask_h, 0.1)

    box_list.append([top_x, top_y, bottom_x-top_x, bottom_y-top_y])

  return box_list

def mask2box_manual(mask):
  # a list to store the x,y,w,h for each bounding box
  box_list = []

  mask_shape = mask.shape
  mask_h = mask_shape[0]
  mask_w = mask_shape[1]

  # all possible values referring to different objects
  possible_values = np.unique(mask)

  # remove the first value cos it is 0
  possible_values = possible_values[1:]

  # the number of all possible values
  num_values = len(possible_values)

  for i in range(num_values):
    # True or False matrix
    bool_matrix = mask == possible_values[i]

    # 0 or 255 matrix
    binary_matrix = np.zeros((mask_h, mask_w)) + np.full((mask_h, mask_w), 255) * bool_matrix
    binary_matrix = binary_matrix.astype(np.uint8)

    r = cv2.selectROI(binary_matrix)
    x = r[0]
    y = r[1]
    w = r[2]
    h = r[3]

    box_list.append([x, y, w, h])

  return box_list

def mask_list2box(mask_list):
  # a list to store the x,y,w,h for each bounding box
  box_list = []

  mask_shape = mask_list[0].shape
  mask_h = mask_shape[0]
  mask_w = mask_shape[1]

  for mask in mask_list:
    mask = mask.astype(np.uint8)
    _, contours = cv2.findContours(mask, 3, 2)

    contour_box_list = []

    for contour_id in range(len(contours)):
      x, y, w, h = cv2.boundingRect(contours[contour_id])
      contour_box_list.append([x, y, x+w, y+h])

    top_x, top_y, bottom_x, bottom_y = generate_ROI(contour_box_list, mask_w, mask_h, 0)

    box_list.append([top_x, top_y, bottom_x, bottom_y])

  return box_list

# separate a label with multiple masks into a list where each element is a mask
def separateMasks(masks):

  mask_list = []

  mask_shape = masks.shape
  mask_h = mask_shape[0]
  mask_w = mask_shape[1]

  # all possible values referring to different objects
  possible_values = np.unique(masks)

  # remove the first value cos it is 0
  possible_values = possible_values[1:]

  # the number of all possible values
  num_values = len(possible_values)

  for i in range(num_values):
    # True or False matrix
    bool_matrix = masks == possible_values[i]

    # 0 or 255 matrix
    binary_matrix = np.zeros((mask_h, mask_w)) + np.full((mask_h, mask_w), 255) * bool_matrix
    binary_matrix = binary_matrix.astype(np.uint8)

    mask_list.append(binary_matrix)

  return possible_values, mask_list

# w,y,w,h to w0,y0,w1,y1
def wh2xy(box):
  return [box[0], box[1], box[0]+box[2], box[1]+box[3]]

# w0,y0,w1,y1 to w,y,w,h
def xy2wh(box):
  return [box[0], box[1], box[2]-box[0], box[3]-box[1]]


# check if one rectangle cover another rectangle
def mat_inter(box1, box2):
  x01, y01, x02, y02 = box1
  x11, y11, x12, y12 = box2

  lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
  ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
  sax = abs(x01 - x02)
  sbx = abs(x11 - x12)
  say = abs(y01 - y02)
  sby = abs(y11 - y12)
  if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
    return True
  else:
    return False

# caculate the IOU of two bounding box
def solve_coincide(box1, box2):
  if mat_inter(box1, box2) == True:
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2
    col = min(x02, x12) - max(x01, x11)
    row = min(y02, y12) - max(y01, y11)
    intersection = col * row
    area1 = (x02 - x01) * (y02 - y01)
    area2 = (x12 - x11) * (y12 - y11)
    coincide = intersection / (area1 + area2 - intersection)
    return coincide
  else:
    return False

def db_eval_iou(annotation,segmentation):
	annotation = annotation.astype(np.bool)
	segmentation = segmentation.astype(np.bool)

	if np.isclose(np.sum(annotation),0) and np.isclose(np.sum(segmentation),0):
		return 1
	else:
		return np.sum((annotation & segmentation)) / \
				np.sum((annotation | segmentation),dtype=np.float32)

# merge the mask to the original frame
def mask_merge(mask, frame):
  original_size = frame.shape
  original_h = original_size[0]
  original_w = original_size[1]
  full = np.full((original_h, original_w, 1), 1)
  bool_mask = mask == 0
  bool_mask = np.reshape(bool_mask, (original_h, original_w, 1))
  mask_frame_2 = np.concatenate((full, bool_mask), axis=2)
  mask_frame_3 = np.concatenate((mask_frame_2, full), axis=2)

  mask_frame = mask_frame_3 * frame
  mask_frame = mask_frame.astype(np.uint8)

  return mask_frame

def prep_display(result_frame_path, dets_out, img, gt, gt_masks, h, w, undo_transform=True, class_color=False):
  """
  Note: If undo_transform=False then im_h and im_w are allowed to be None.
  gt and gt_masks are also allowed to be none (until I reimplement that functionality).
  """
  if undo_transform:
    img_numpy = undo_image_transformation(img, w, h)
    img_gpu = torch.Tensor(img_numpy).cuda()
  else:
    img_gpu = img / 255.0
    h, w, _ = img.shape

  with timer.env('Postprocess'):
    t = postprocess(dets_out, w, h, visualize_lincomb=args.display_lincomb, crop_masks=args.crop, score_threshold=args.score_threshold)
    torch.cuda.synchronize()

  with timer.env('Copy'):
    if cfg.eval_mask_branch:
      masks = t[3][:args.top_k] # We'll need this later
    classes, scores, boxes = [x[:args.top_k].cpu().numpy() for x in t[:3]]

  if classes.shape[0] == 0:
    return (img_gpu * 255).byte().cpu().numpy()

  def get_color(j):
    color = COLORS[(classes[j] * 5 if class_color else j * 5) % len(COLORS)]
    if not undo_transform:
      color = (color[2], color[1], color[0])
    return color

  # Draw masks first on the gpu
  if args.display_masks and cfg.eval_mask_branch:
    for j in reversed(range(min(args.top_k, classes.shape[0]))):
      #if scores[j] >= args.score_threshold:
      if scores[j] >= 0:
        color = get_color(j)

        mask = masks[j, :, :, None]
        mask_color = mask @ (torch.Tensor(color).view(1, 3) / 255.0)
        mask_alpha = 0.45

        # Alpha only the region of the image that contains the mask
        img_gpu = img_gpu * (1 - mask) \
                  + img_gpu * mask * (1-mask_alpha) + mask_color * mask_alpha

  # Then draw the stuff that needs to be done on the cpu
  # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
  img_numpy = (img_gpu * 255).byte().cpu().numpy()

  if args.display_text or args.display_bboxes:
    for j in reversed(range(min(args.top_k, classes.shape[0]))):
      score = scores[j]

      #if scores[j] >= args.score_threshold:
      if scores[j] >= 0:
        x1, y1, x2, y2 = boxes[j, :]
        color = get_color(j)

        if args.display_bboxes:
          cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

        if args.display_text:
          _class = COCO_CLASSES[classes[j]]
          text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class

          font_face = cv2.FONT_HERSHEY_DUPLEX
          font_scale = 0.6
          font_thickness = 1

          text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

          text_pt = (x1, y1 - 3)
          text_color = [255, 255, 255]

          cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
          cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

  return img_numpy, t

def prep_benchmark(dets_out, h, w):
  with timer.env('Postprocess'):
    t = postprocess(dets_out, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)

  with timer.env('Copy'):
    classes, scores, boxes, masks = [x[:args.top_k].cpu().numpy() for x in t]

  with timer.env('Sync'):
    # Just in case
    torch.cuda.synchronize()

def prep_coco_cats(cats):
  """ Prepare inverted table for category id lookup given a coco cats object. """
  name_lookup = {}

  for _id, cat_obj in cats.items():
    name_lookup[cat_obj['name']] = _id

  # Bit of a roundabout way to do this but whatever
  for i in range(len(COCO_CLASSES)):
    coco_cats.append(name_lookup[COCO_CLASSES[i]])
    coco_cats_inv[coco_cats[-1]] = i


def get_coco_cat(transformed_cat_id):
  """ transformed_cat_id is [0,80) as indices in COCO_CLASSES """
  return coco_cats[transformed_cat_id]

def get_transformed_cat(coco_cat_id):
  """ transformed_cat_id is [0,80) as indices in COCO_CLASSES """
  return coco_cats_inv[coco_cat_id]


class Detections:

  def __init__(self):
    self.bbox_data = []
    self.mask_data = []

  def add_bbox(self, image_id:int, category_id:int, bbox:list, score:float):
    """ Note that bbox should be a list or tuple of (x1, y1, x2, y2) """
    bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

    # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
    bbox = [round(float(x)*10)/10 for x in bbox]

    self.bbox_data.append({
      'image_id': int(image_id),
      'category_id': get_coco_cat(int(category_id)),
      'bbox': bbox,
      'score': float(score)
    })

  def add_mask(self, image_id:int, category_id:int, segmentation:np.ndarray, score:float):
    """ The segmentation should be the full mask, the size of the image and with size [h, w]. """
    rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('ascii') # json.dump doesn't like bytes strings

    self.mask_data.append({
      'image_id': int(image_id),
      'category_id': get_coco_cat(int(category_id)),
      'segmentation': rle,
      'score': float(score)
    })

  def dump(self):
    dump_arguments = [
      (self.bbox_data, args.bbox_det_file),
      (self.mask_data, args.mask_det_file)
    ]

    for data, path in dump_arguments:
      with open(path, 'w') as f:
        json.dump(data, f)

  def dump_web(self):
    """ Dumps it in the format for my web app. Warning: bad code ahead! """
    config_outs = ['preserve_aspect_ratio', 'use_prediction_module',
                   'use_yolo_regressors', 'use_prediction_matching',
                   'train_masks']

    output = {
      'info' : {
        'Config': {key: getattr(cfg, key) for key in config_outs},
      }
    }

    image_ids = list(set([x['image_id'] for x in self.bbox_data]))
    image_ids.sort()
    image_lookup = {_id: idx for idx, _id in enumerate(image_ids)}

    output['images'] = [{'image_id': image_id, 'dets': []} for image_id in image_ids]

    # These should already be sorted by score with the way prep_metrics works.
    for bbox, mask in zip(self.bbox_data, self.mask_data):
      image_obj = output['images'][image_lookup[bbox['image_id']]]
      image_obj['dets'].append({
        'score': bbox['score'],
        'bbox': bbox['bbox'],
        'category': COCO_CLASSES[get_transformed_cat(bbox['category_id'])],
        'mask': mask['segmentation'],
      })

    with open(os.path.join(args.web_det_path, '%s.json' % cfg.name), 'w') as f:
      json.dump(output, f)




def mask_iou(mask1, mask2, iscrowd=False):
  """
  Inputs inputs are matricies of size _ x N. Output is size _1 x _2.
  Note: if iscrowd is True, then mask2 should be the crowd.
  """
  timer.start('Mask IoU')

  intersection = torch.matmul(mask1, mask2.t())
  area1 = torch.sum(mask1, dim=1).view(1, -1)
  area2 = torch.sum(mask2, dim=1).view(1, -1)
  union = (area1.t() + area2) - intersection

  if iscrowd:
    # Make sure to brodcast to the right dimension
    ret = intersection / area1.t()
  else:
    ret = intersection / union
  timer.stop('Mask IoU')
  return ret.cpu()

def bbox_iou(bbox1, bbox2, iscrowd=False):
  with timer.env('BBox IoU'):
    ret = jaccard(bbox1, bbox2, iscrowd)
  return ret.cpu()

def prep_metrics(ap_data, dets, img, gt, gt_masks, h, w, num_crowd, image_id, detections:Detections=None):
  """ Returns a list of APs for this image, with each element being for a class  """
  if not args.output_coco_json:
    with timer.env('Prepare gt'):
      gt_boxes = torch.Tensor(gt[:, :4])
      gt_boxes[:, [0, 2]] *= w
      gt_boxes[:, [1, 3]] *= h
      gt_classes = list(gt[:, 4].astype(int))
      gt_masks = torch.Tensor(gt_masks).view(-1, h*w)

      if num_crowd > 0:
        split = lambda x: (x[-num_crowd:], x[:-num_crowd])
        crowd_boxes  , gt_boxes   = split(gt_boxes)
        crowd_masks  , gt_masks   = split(gt_masks)
        crowd_classes, gt_classes = split(gt_classes)

  with timer.env('Postprocess'):
    classes, scores, boxes, masks = postprocess(dets, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)

    if classes.size(0) == 0:
      return

    classes = list(classes.cpu().numpy().astype(int))
    scores = list(scores.cpu().numpy().astype(float))
    masks = masks.view(-1, h*w).cuda()
    boxes = boxes.cuda()


  if args.output_coco_json:
    with timer.env('JSON Output'):
      boxes = boxes.cpu().numpy()
      masks = masks.view(-1, h, w).cpu().numpy()
      for i in range(masks.shape[0]):
        # Make sure that the bounding box actually makes sense and a mask was produced
        if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:
          detections.add_bbox(image_id, classes[i], boxes[i,:],   scores[i])
          detections.add_mask(image_id, classes[i], masks[i,:,:], scores[i])
      return

  with timer.env('Eval Setup'):
    num_pred = len(classes)
    num_gt   = len(gt_classes)

    mask_iou_cache = mask_iou(masks, gt_masks)
    bbox_iou_cache = bbox_iou(boxes.float(), gt_boxes.float())

    if num_crowd > 0:
      crowd_mask_iou_cache = mask_iou(masks, crowd_masks, iscrowd=True)
      crowd_bbox_iou_cache = bbox_iou(boxes.float(), crowd_boxes.float(), iscrowd=True)
    else:
      crowd_mask_iou_cache = None
      crowd_bbox_iou_cache = None

    iou_types = [
      ('box',  lambda i,j: bbox_iou_cache[i, j].item(), lambda i,j: crowd_bbox_iou_cache[i,j].item()),
      ('mask', lambda i,j: mask_iou_cache[i, j].item(), lambda i,j: crowd_mask_iou_cache[i,j].item())
    ]

  timer.start('Main loop')
  for _class in set(classes + gt_classes):
    ap_per_iou = []
    num_gt_for_class = sum([1 for x in gt_classes if x == _class])

    for iouIdx in range(len(iou_thresholds)):
      iou_threshold = iou_thresholds[iouIdx]

      for iou_type, iou_func, crowd_func in iou_types:
        gt_used = [False] * len(gt_classes)

        ap_obj = ap_data[iou_type][iouIdx][_class]
        ap_obj.add_gt_positives(num_gt_for_class)

        for i in range(num_pred):
          if classes[i] != _class:
            continue

          max_iou_found = iou_threshold
          max_match_idx = -1
          for j in range(num_gt):
            if gt_used[j] or gt_classes[j] != _class:
              continue

            iou = iou_func(i, j)

            if iou > max_iou_found:
              max_iou_found = iou
              max_match_idx = j

          if max_match_idx >= 0:
            gt_used[max_match_idx] = True
            ap_obj.push(scores[i], True)
          else:
            # If the detection matches a crowd, we can just ignore it
            matched_crowd = False

            if num_crowd > 0:
              for j in range(len(crowd_classes)):
                if crowd_classes[j] != _class:
                  continue

                iou = crowd_func(i, j)

                if iou > iou_threshold:
                  matched_crowd = True
                  break

            # All this crowd code so that we can make sure that our eval code gives the
            # same result as COCOEval. There aren't even that many crowd annotations to
            # begin with, but accuracy is of the utmost importance.
            if not matched_crowd:
              ap_obj.push(scores[i], False)
  timer.stop('Main loop')

def BGR2RGB(frame_BGR, w, h):

  R_channel = frame_BGR[:, :, 2].copy()
  G_channel = frame_BGR[:, :, 1].copy()
  B_channel = frame_BGR[:, :, 0].copy()

  frame_BGR_size = frame_BGR.shape
  h = frame_BGR_size[0]
  w = frame_BGR_size[1]

  frame_RGB = np.zeros((h, w, 3))

  frame_RGB[:, :, 0] = R_channel
  frame_RGB[:, :, 1] = G_channel
  frame_RGB[:, :, 2] = B_channel

  return frame_RGB

def generate_state(ROI_box_last_target, ROI_box_last_frame, ROI_max_iou_mask, cropped_frame, w, h, normalization, extract_model):


  '''
  frame_copy = cropped_frame.copy()
  frame_copy = mask_merge(ROI_max_iou_mask, frame_copy)
  cv2.imshow("ROI_Box_last_frame", frame_copy)
  cv2.waitKey(0)
  '''

  '''
  frame_copy = ROI_box_last_frame.copy()
  frame_copy = cv2.rectangle(frame_copy, (ROI_box_last_target[0], ROI_box_last_target[1]), (ROI_box_last_target[2], ROI_box_last_target[3]), (255, 0, 0), 2)
  cv2.imshow("ROI_Box_last_frame", frame_copy)
  cv2.waitKey(0)
  '''

  frame_BGR_size = ROI_box_last_frame.shape
  h = frame_BGR_size[0]
  w = frame_BGR_size[1]

  ROI_box_last_frame = BGR2RGB(ROI_box_last_frame, w, h)
  cropped_frame = BGR2RGB(cropped_frame, w, h)



  # generate the picture(RGB) of the frame with the last "correct bounding box"
  ROI_full_zero = np.zeros((h, w, 3))
  ROI_full_zero = ROI_full_zero.astype(np.uint8)

  # ROI_box_last_target is the sigle last "correct bounding box" of a certain target
  # ROI_box_last_frame is its corresponding frame
  ROI_full_zero[ROI_box_last_target[1]:ROI_box_last_target[3], ROI_box_last_target[0]:ROI_box_last_target[2], :] \
    = ROI_box_last_frame[ROI_box_last_target[1]:ROI_box_last_target[3], ROI_box_last_target[0]:ROI_box_last_target[2], :]

  ROI_box_cropped_frame = ROI_full_zero.copy()

  # generate the picture(RGB) of the frame with the current mask predicted
  # cropped_frame is a cropped patch of current frame
  ROI_mask_cropped_frame = cropped_frame.copy()

  ROI_max_iou_mask = ROI_max_iou_mask / 255

  # only show the area of the mask
  ROI_mask_cropped_frame[:, :, 0] = ROI_mask_cropped_frame[:, :, 0] * ROI_max_iou_mask
  ROI_mask_cropped_frame[:, :, 1] = ROI_mask_cropped_frame[:, :, 1] * ROI_max_iou_mask
  ROI_mask_cropped_frame[:, :, 2] = ROI_mask_cropped_frame[:, :, 2] * ROI_max_iou_mask

  # convert the numpy array to PIL
  ROI_box_cropped_frame = ROI_box_cropped_frame.astype(np.uint8)
  ROI_box_cropped_frame = Image.fromarray(ROI_box_cropped_frame)
  ROI_mask_cropped_frame = ROI_mask_cropped_frame.astype(np.uint8)
  ROI_mask_cropped_frame = Image.fromarray(ROI_mask_cropped_frame)

  #ROI_box_cropped_frame.show()
  #ROI_mask_cropped_frame.show()

  # generate the feature of the frame with the last "correct bounding box"
  ROI_box_cropped_frame = normalization(ROI_box_cropped_frame)
  ROI_box_cropped_frame.unsqueeze_(dim=0)
  ROI_box_cropped_frame_input = copy.deepcopy(ROI_box_cropped_frame)
  ROI_box_cropped_frame_input = np.array(ROI_box_cropped_frame_input)
  ROI_box_cropped_frame_input = torch.Tensor(ROI_box_cropped_frame_input).cuda().float()

  ROI_box_cropped_frame_output = extract_model.conv1(ROI_box_cropped_frame_input)
  ROI_box_cropped_frame_output = extract_model.bn1(ROI_box_cropped_frame_output)
  ROI_box_cropped_frame_output = extract_model.relu(ROI_box_cropped_frame_output)
  ROI_box_cropped_frame_output = extract_model.maxpool(ROI_box_cropped_frame_output)
  ROI_box_cropped_frame_output = extract_model.layer1(ROI_box_cropped_frame_output)
  ROI_box_cropped_frame_output = extract_model.layer2(ROI_box_cropped_frame_output)
  ROI_box_cropped_frame_output = extract_model.layer3(ROI_box_cropped_frame_output)
  ROI_box_cropped_frame_output = extract_model.layer4(ROI_box_cropped_frame_output)
  ROI_box_cropped_frame_state = extract_model.avgpool(ROI_box_cropped_frame_output)

  ROI_box_cropped_frame_state = ROI_box_cropped_frame_state.reshape(-1).detach()

  # generate the feature of the frame with the current mask predicted
  ROI_mask_cropped_frame = normalization(ROI_mask_cropped_frame)
  ROI_mask_cropped_frame.unsqueeze_(dim=0)
  ROI_mask_cropped_frame_input = copy.deepcopy(ROI_mask_cropped_frame)
  ROI_mask_cropped_frame_input = np.array(ROI_mask_cropped_frame_input)
  ROI_mask_cropped_frame_input = torch.Tensor(ROI_mask_cropped_frame_input).cuda().float()

  ROI_mask_cropped_frame_output = extract_model.conv1(ROI_mask_cropped_frame_input)
  ROI_mask_cropped_frame_output = extract_model.bn1(ROI_mask_cropped_frame_output)
  ROI_mask_cropped_frame_output = extract_model.relu(ROI_mask_cropped_frame_output)
  ROI_mask_cropped_frame_output = extract_model.maxpool(ROI_mask_cropped_frame_output)
  ROI_mask_cropped_frame_output = extract_model.layer1(ROI_mask_cropped_frame_output)
  ROI_mask_cropped_frame_output = extract_model.layer2(ROI_mask_cropped_frame_output)
  ROI_mask_cropped_frame_output = extract_model.layer3(ROI_mask_cropped_frame_output)
  ROI_mask_cropped_frame_output = extract_model.layer4(ROI_mask_cropped_frame_output)
  ROI_mask_cropped_frame_state = extract_model.avgpool(ROI_mask_cropped_frame_output)

  ROI_mask_cropped_frame_state = ROI_mask_cropped_frame_state.reshape(-1).detach()

  # concatenate these two feature together
  state = torch.cat((ROI_mask_cropped_frame_state, ROI_box_cropped_frame_state), 0)

  return state

def generate_state_2(ROI_max_iou_mask, cropped_frame, normalization, resnet50):

  extract_model = resnet50

  frame_BGR_size = cropped_frame.shape
  h = frame_BGR_size[0]
  w = frame_BGR_size[1]

  cropped_frame = BGR2RGB(cropped_frame, w, h)

  cropped_frame_mask = cropped_frame.copy()
  cropped_frame_mask[:, :, 0] = cropped_frame[:, :, 0] * ROI_max_iou_mask
  cropped_frame_mask[:, :, 1] = cropped_frame[:, :, 1] * ROI_max_iou_mask
  cropped_frame_mask[:, :, 2] = cropped_frame[:, :, 2] * ROI_max_iou_mask

  # convert the numpy array to PIL
  cropped_frame = cropped_frame.astype(np.uint8)
  cropped_frame = Image.fromarray(cropped_frame)

  cropped_frame_mask = cropped_frame_mask.astype(np.uint8)
  cropped_frame_mask = Image.fromarray(cropped_frame_mask)

  # generate the feature of the cropped frame
  cropped_frame = normalization(cropped_frame)
  cropped_frame.unsqueeze_(dim=0)
  cropped_frame_input = copy.deepcopy(cropped_frame)
  cropped_frame_input = np.array(cropped_frame_input)
  cropped_frame_input = torch.Tensor(cropped_frame_input).cuda().float()

  cropped_frame_output = extract_model.conv1(cropped_frame_input)
  cropped_frame_output = extract_model.bn1(cropped_frame_output)
  cropped_frame_output = extract_model.relu(cropped_frame_output)
  cropped_frame_output = extract_model.maxpool(cropped_frame_output)
  cropped_frame_output = extract_model.layer1(cropped_frame_output)
  cropped_frame_output = extract_model.layer2(cropped_frame_output)
  cropped_frame_output = extract_model.layer3(cropped_frame_output)
  cropped_frame_output = extract_model.layer4(cropped_frame_output)
  cropped_frame_state = extract_model.avgpool(cropped_frame_output)

  cropped_frame_state = cropped_frame_state.reshape(-1).detach()

  # generate the feature of the cropped frame only with mask
  cropped_frame_mask = normalization(cropped_frame_mask)
  cropped_frame_mask.unsqueeze_(dim=0)
  cropped_frame_input_mask = copy.deepcopy(cropped_frame_mask)
  cropped_frame_input_mask = np.array(cropped_frame_input_mask)
  cropped_frame_input_mask = torch.Tensor(cropped_frame_input_mask).cuda().float()

  cropped_frame_output_mask = extract_model.conv1(cropped_frame_input_mask)
  cropped_frame_output_mask = extract_model.bn1(cropped_frame_output_mask)
  cropped_frame_output_mask = extract_model.relu(cropped_frame_output_mask)
  cropped_frame_output_mask = extract_model.maxpool(cropped_frame_output_mask)
  cropped_frame_output_mask = extract_model.layer1(cropped_frame_output_mask)
  cropped_frame_output_mask = extract_model.layer2(cropped_frame_output_mask)
  cropped_frame_output_mask = extract_model.layer3(cropped_frame_output_mask)
  cropped_frame_output_mask = extract_model.layer4(cropped_frame_output_mask)
  cropped_frame_state_mask = extract_model.avgpool(cropped_frame_output_mask)

  cropped_frame_mask_state = cropped_frame_state.reshape(-1).detach()

  # concatenate these two feature together
  state = torch.cat((cropped_frame_state, cropped_frame_mask_state), 0)

  return state

  pass




class APDataObject:
  """
  Stores all the information necessary to calculate the AP for one IoU and one class.
  Note: I type annotated this because why not.
  """

  def __init__(self):
    self.data_points = []
    self.num_gt_positives = 0

  def push(self, score:float, is_true:bool):
    self.data_points.append((score, is_true))

  def add_gt_positives(self, num_positives:int):
    """ Call this once per image. """
    self.num_gt_positives += num_positives

  def is_empty(self) -> bool:
    return len(self.data_points) == 0 and self.num_gt_positives == 0

  def get_ap(self) -> float:
    """ Warning: result not cached. """

    if self.num_gt_positives == 0:
      return 0

    # Sort descending by score
    self.data_points.sort(key=lambda x: -x[0])

    precisions = []
    recalls = []
    num_true = 0
    num_false = 0

    # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
    for datum in self.data_points:
      # datum[1] is whether the detection a true or false positive
      if datum[1]: num_true += 1
      else: num_false += 1

      precision = num_true / (num_true + num_false)
      recall = num_true / self.num_gt_positives

      precisions.append(precision)
      recalls.append(recall)

    # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
    # Basically, remove any temporary dips from the curve.
    # At least that's what I think, idk. COCOEval did it so I do too.
    for i in range(len(precisions)-1, 0, -1):
      if precisions[i] > precisions[i-1]:
        precisions[i-1] = precisions[i]

    # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
    y_range = [0] * 101 # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
    x_range = np.array([x / 100 for x in range(101)])
    recalls = np.array(recalls)

    # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
    # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
    # I approximate the integral this way, because that's how COCOEval does it.
    indices = np.searchsorted(recalls, x_range, side='left')
    for bar_idx, precision_idx in enumerate(indices):
      if precision_idx < len(precisions):
        y_range[bar_idx] = precisions[precision_idx]

    # Finally compute the riemann sum to get our integral.
    # avg([precision(x) for x in 0:0.01:1])
    return sum(y_range) / len(y_range)

class Actor(nn.Module):
  def __init__(self, state_size, action_size):
    super(Actor, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.linear1 = nn.Linear(self.state_size, 2048)
    self.linear2 = nn.Linear(2048, 512)
    self.linear3 = nn.Linear(512, self.action_size)

  def forward(self, state):
    output = F.relu(self.linear1(state))
    output = F.relu(self.linear2(output))
    output = self.linear3(output)
    distribution = Categorical(F.softmax(output, dim=-1))
    return distribution


class Critic(nn.Module):
  def __init__(self, state_size, action_size):
    super(Critic, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.linear1 = nn.Linear(self.state_size, 2048)
    self.linear2 = nn.Linear(2048, 512)
    self.linear3 = nn.Linear(512, 1)

  def forward(self, state):
    output = F.relu(self.linear1(state))
    output = F.relu(self.linear2(output))
    value = self.linear3(output)
    return value

def compute_returns(next_value, rewards, masks, gamma=0.99):
  R = next_value
  returns = []
  for step in reversed(range(len(rewards))):
    R = rewards[step] + gamma * R * masks[step]
    returns.insert(0, R)
  return returns

def badhash(x):
  """
  Just a quick and dirty hash function for doing a deterministic shuffle based on image_id.

  Source:
  https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
  """
  x = (((x >> 16) ^ x) * 0x45d9f3b) & 0xFFFFFFFF
  x = (((x >> 16) ^ x) * 0x45d9f3b) & 0xFFFFFFFF
  x = ((x >> 16) ^ x) & 0xFFFFFFFF
  return x


#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################



def evalimage(net:Yolact, path:str, save_path:str=None):

  '''
  # state_size = 25088  # the size of the state after flatten
  state_size = 4096  # the size of the state after flatten
  action_size = 2  # the number of actions of actor
  e_iters = 20000  # the number of training iteration
  save_iters = 100  # the frequency of saving a model
  lr_decay_iters = 20000  # the frequency of decay the learning rate
  actor_lr = 0.000001  # learning rate of actor model
  critic_lr = 0.000005  # learning rate of critic model
  first_train = True  # whether initialize a new model or load a existing model
  vgg_layer = 30  # the output layer index of pre-trained vgg model
  save_path_actor = "./weights/RL_model/actor.pkl"
  save_path_critic = "./weights/RL_model/critic.pkl"
  '''

  torch.manual_seed(1)
  random.seed(1)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  actor = Actor(state_size, action_size).cuda()
  critic = Critic(state_size, action_size).cuda()
  optimizerA = optim.Adam(actor.parameters(), lr=actor_lr)
  optimizerC = optim.Adam(critic.parameters(), lr=critic_lr)

  resnet50 = models.resnet50(pretrained=True).cuda()
  resnet50.eval()

  normalization = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
  ])

  #### modified part #####

  # the path of the image
  dataset_path = os.path.join(dataset_root, "JPEGImages", "480p")

  # the path of the label
  label_path = os.path.join(dataset_root, "Annotations", "480p")

  # the path to save the result
  result_root_path = "./results"

  # import all evaluation sequences
  eval_sequence_list_path = os.path.join(dataset_root, "ImageSets", "2017", "val.txt")
  train_sequence_list_path = os.path.join(dataset_root, "ImageSets", "2017", "train.txt")

  # import all sequecnes 
  f = open(eval_sequence_list_path, "r")
  eval_sequence_list = f.read().splitlines()
  f.close()

  f = open(train_sequence_list_path, "r")
  train_sequence_list = f.read().splitlines()
  f.close()

  best_accuracy = 0

  eval_count = 1

  f = open("./result", "w")
  f.close()

  flog = open("./log", "w")
  flog.close()

  e = -1

  while True:
    try:
      e += 1
      total_accuracy = 0
      eval_count += 1

      if (eval_count%2) == 0:
        sequence_list = train_sequence_list
      else:
        sequence_list = eval_sequence_list

      for sequence in sequence_list:
        if sequence in region_ratio_list:
          search_region_ration = region_ratio_list[sequence]
        else:
          search_region_ration = 0.4

        #### #### directory to save result
        # create a directory for this sequence to store mask results of this sequence
        result_sequence_path = os.path.join(result_root_path, sequence)
        # check if this directory already exist first before generate this directory
        if not os.path.isdir(result_sequence_path):
          os.mkdir(result_sequence_path)

        ####### import the label of the first frame (frame 0) and its label bounding box
        lable_sequence_path = os.path.join(label_path, sequence)
        label_first_path = os.path.join(lable_sequence_path, "00000.png")
        first_label = cv2.imread(label_first_path, cv2.IMREAD_GRAYSCALE)

        # the original size of the frame
        original_size = first_label.shape
        original_h = original_size[0]
        original_w = original_size[1]

        # the bounding box list for the previous frame (now is the instance masks of frame 0), four elements indicate x, y, w, h, respectively
        full_box_list_last = mask2box(first_label)
        #box_list_last = mask2box_manual(first_label)

        # each element in mask_list_last is a mask (two channel, 0, 255)
        target_values, full_mask_list_last = separateMasks(first_label)

        # the list to store all training-needed data for each target
        train_data = []

        # convert all elements in box_list_last from w,h to x2,y2
        for i in range(len(full_box_list_last)):
          full_box_list_last[i] = wh2xy(full_box_list_last[i])

          # each element for:
          # 0: log_prob_list
          # 1: value_list
          # 2: reward_list
          # 3: done_list
          train_data.append([[], [], [], [], [], []])

        '''
        # show indivial smaller bounding box of each tagert
        for i in range(len(box_list_last)):
          cv2.rectangle(first_label, (box_list_last[i][0], box_list_last[i][1]), (box_list_last[i][2], box_list_last[i][3]), 255)
          cv2.imshow("first_box_test", first_label)
          cv2.waitKey(0)
        '''

        # generate the ROI of the bounding box of all smaller bounding boxes
        top_x, top_y, bottom_x, bottom_y = generate_ROI(full_box_list_last, original_w, original_h, 0.1)

        '''
        # check the big bounding box of all smaller bounding box of ground truth
        cv2.rectangle(first_label, (top_x, top_y), (bottom_x, bottom_y), (255,0,0), 2)
        cv2.imshow("0", first_label)
        cv2.waitKey(0)
        '''

        ####### import the image since frame 1 (first frame need to be forwarded)
        sequence_directory_path = os.path.join(dataset_path, sequence)
        frame_name_list = os.listdir(sequence_directory_path)
        frame_name_list.sort()

        # remove frame 0
        frame_name_0 = frame_name_list.pop(0)
        frame_path_0 = os.path.join(sequence_directory_path, frame_name_0)
        correct_last_frame = cv2.imread(frame_path_0)
        ROI_correct_last_frame = correct_last_frame[top_y:bottom_y, top_x:bottom_x, :]

        # used to test since a particular frame
        '''
        for i in range(95):
          frame_name_list.pop(0)
        '''

        for frame_name_id in range(len(frame_name_list)):
        #for frame_name_id in range(frame_strart):

          frame_name = frame_name_list[frame_name_id]
          frame_path = os.path.join(sequence_directory_path, frame_name)
          frame_numpy = cv2.imread(frame_path)

          label_t_name = frame_name.split(".")[0] + ".png"
          label_t_path = os.path.join(lable_sequence_path, label_t_name)
          label_t_numpy = cv2.imread(label_t_path, cv2.IMREAD_GRAYSCALE)

          pred_values, label_t_mask_full_list = mask_divided(label_t_numpy)

          label_t_box_full_list = mask2box(label_t_numpy)

          ####### choose the ROI (repalce the original ROI function of cv2)
          x = top_x
          y = top_y
          w = bottom_x - top_x
          h = bottom_y - top_y

          cropped_frame = frame_numpy[y:y+h, x:x+w, :]
          cropped_frame_tensor = torch.Tensor(cropped_frame).cuda().float()

          #####################################
          ###### original code as follows: ####
          #####################################
          batch = FastBaseTransform()(cropped_frame_tensor.unsqueeze(0))

          with torch.no_grad():
            preds = net(batch)

          '''
          # show all candidate bounding boxes and its probability
          candidate_box_list = preds[0]["box"]
          candidate_box_list = candidate_box_list.cpu().numpy()
          score_list = preds[0]["score"]
          score_list = score_list.cpu().numpy()
          for i in range(score_list.shape[0]):
            candidate_box = candidate_box_list[i]
            cropped_frame_copy = cropped_frame.copy()
            cv2.rectangle(cropped_frame_copy, (int(candidate_box[0]*w), int(candidate_box[1]*h)), (int(candidate_box[2]*w), int(candidate_box[3]*h)), (255,0,0), 2)
            cv2.imshow("1", cropped_frame_copy)
            cv2.waitKey(0)
            print(str(i) + " " + str(score_list[i]))
          cv2.destroyAllWindows()
          '''

          '''
          # show the content of all 32 prototypes
          prototype_list = preds[0]["proto"]
          prototype_list = prototype_list.cpu().numpy()

          for i in range(32):
            cv2.imshow(str(i), prototype_list[:,:,i])
          cv2.waitKey(0)
          '''
          result_frame_path = os.path.join(result_sequence_path, frame_name.split(".")[0])
          # the command to actually process this frame (cropped)
          # t is actually the final result
          # t[0] is the id of the category of each predicted instance
          # t[1] is the probability value of each predicted instace
          # t[2] is the bounding box of each predicted instance (x1,y2,x2,y2)
          # t[3] is the mask of each precited instance

          try:
            img_numpy, t = prep_display(result_frame_path, preds, cropped_frame_tensor, None, None, None, None, undo_transform=False,)
          except ValueError:
            break
          else:
            pass

          # all these data are for prediction result

          id_list_t = t[0].cpu().numpy()
          probability_list_t = t[1].cpu().numpy()
          ROI_box_list_t_numpy = t[2].cpu().numpy()
          ROI_box_list_t = []

          id_list_t_len = id_list_t.size

          for i in range(id_list_t_len):
            ROI_box_list_t.append(ROI_box_list_t_numpy[i])

          ROI_mask_list_t_numpy = t[3].cpu().numpy()
          ROI_mask_list_t = []

          for i in range(id_list_t_len):
            ROI_mask_list_t.append(ROI_mask_list_t_numpy[i])
          # generate the number of all instances which survive
          instance_num = id_list_t.shape[0]

          '''
          ###### code for visualization (original code) ###########
          if save_path is None:
            img_numpy = img_numpy[:, :, (2, 1, 0)]
          
          if save_path is None:
            plt.imshow(img_numpy)
            plt.title(path)
            plt.show()
          else:
            cv2.imwrite(save_path, img_numpy)
          #########################################
          '''

          ###### restore the cropped mask and bounding box to original size #######
          # full_mask_list is used to store the mask with the original size
          full_mask_list_t = []
          # full_box_list is used to store the bounding box with the original size
          # x1, y1, x2, y2
          full_box_list_t = []

          # instacnce_num is the number of all prediction result
          for i in range(instance_num):
            # process the mask
            cropped_mask = ROI_mask_list_t[i]
            full_mask = np.zeros((original_h, original_w))
            full_mask[y:y+h, x:x+w] = cropped_mask
            # show the full mask
            #cv2.imshow("full_mask_test", full_mask)
            #cv2.waitKey(0)
            # original value for foreground is 1.0, turns it to 255.0 now
            full_mask_list_t.append(full_mask*255)

            # process the bounding box
            full_box = [ROI_box_list_t[i][0]+x, ROI_box_list_t[i][1]+y, ROI_box_list_t[i][2]+x, ROI_box_list_t[i][3]+y]
            # test the full bounding box
            #cv2.rectangle(frame_numpy, (full_box[0], full_box[1]), (full_box[2], full_box[3]), 255)
            #cv2.imshow("full_box_test", frame_numpy)
            #cv2.waitKey(0)
            full_box_list_t.append(full_box)

          # convert the box in full_box_list_last to ROI (smaller)
          ROI_box_list_last = []
          for full_box_last in full_box_list_last:
            ROI_box_last = [full_box_last[0] - x, full_box_last[1] - y, full_box_last[2] - x, full_box_last[3] - y]
            ROI_box_list_last.append(ROI_box_last)

          # for each previous bounding box, find the bounding box with the highest IOU in the current frame
          # box_list_last is the correct location bounding box for each target
          # box_last_id is the index

          # for each target in the video
          for target_last_id in range(len(full_box_list_last)):

            # each element in iou_list is the iou between the box_last and a certain box_t
            # for each candidate prediction result
            box_iou_list = []
            mask_iou_list = []

            # instacnce_num is the number of all prediction result
            for prediction_t_id in range(instance_num):
              frame_copy = frame_numpy.copy()
              predicted_box = full_box_list_t[prediction_t_id]
              box_iou = solve_coincide(full_box_list_last[target_last_id], predicted_box)
              if box_iou == False:
                box_iou = 0
              box_iou_list.append(box_iou)

              predicted_mask = full_mask_list_t[prediction_t_id]
              mask_iou = db_eval_iou(full_mask_list_last[target_last_id], predicted_mask)
              mask_iou_list.append(mask_iou)

              '''
              # check the bounding box and it corresponding iou, just for test
              cv2.rectangle(frame_copy, (predicted_box[0], predicted_box[1]), (predicted_box[2], predicted_box[3]), (0,0,255), 2)
              cv2.imshow("", frame_copy)
              cv2.waitKey(0)
              #print("iou: " + str(iou))
              '''

              # id for iou_list. full_box_list, mask_list are corresponded
              box_iou_list_copy = box_iou_list.copy()
              mask_iou_list_copy = mask_iou_list.copy()

              full_box_list_t_copy = full_box_list_t.copy()
              full_mask_list_t_copy = full_mask_list_t.copy()

              ROI_box_list_t_copy = ROI_box_list_t.copy()
              ROI_mask_list_t_copy = ROI_mask_list_t.copy()

              #last_max_iou_t = 0
              #last_max_mask_iou_t = 0

            # find the value of the maximum iou (most proper) in current iou_list
            max_box_iou_t = max(box_iou_list_copy)
            max_mask_iou_t = max(mask_iou_list_copy)

            # find the index of the maximum iou in current iou_list
            max_box_iou_index_t = box_iou_list_copy.index(max_box_iou_t)
            max_mask_iou_index_t = mask_iou_list_copy.index(max_mask_iou_t)

            '''
            # remove all duplicated data which is idendical to this candidate result
            if max_box_iou_t == last_max_iou_t:
              box_iou_list_copy.pop(max_box_iou_index_t)
              full_box_list_t_copy.pop(max_box_iou_index_t)
              ROI_box_list_t_copy.pop(max_box_iou_index_t)
              full_mask_list_t_copy.pop(max_box_iou_index_t)
              ROI_mask_list_t_copy.pop(max_box_iou_index_t)
              continue
            '''

            # only used to remove duplicated data
            #last_max_iou_t = max_box_iou_t

            # get the cooresponding box
            full_max_iou_box = full_box_list_t_copy[max_box_iou_index_t]
            ROI_max_iou_box = ROI_box_list_t_copy[max_box_iou_index_t]

            # get the cooresponding mask
            full_max_iou_mask = full_mask_list_t_copy[max_mask_iou_index_t]
            ROI_max_iou_mask = ROI_mask_list_t_copy[max_mask_iou_index_t]

            # get the last "correct bounding box"
            ROI_box_last_target = ROI_box_list_last[target_last_id]
            full_box_last_target = full_box_list_last[target_last_id]

            # show the mask on the frame
            full_mask_frame = mask_merge(full_max_iou_mask, frame_numpy)
            ROI_mask_frame = mask_merge(ROI_max_iou_mask, cropped_frame)

            # add the bounding box on the frame
            #cv2.rectangle(full_mask_frame, (full_max_iou_box[0], full_max_iou_box[1]), (full_max_iou_box[2], full_max_iou_box[3]), (255, 0, 0), 2)
            #cv2.rectangle(full_mask_frame, (full_box_list_last[box_last_id][0], full_box_list_last[box_last_id][1]), (full_box_list_last[box_last_id][2], full_box_list_last[box_last_id][3]), (0,255,0), 2)
            #cv2.rectangle(ROI_mask_frame, (ROI_max_iou_box[0], ROI_max_iou_box[1]), (ROI_max_iou_box[2], ROI_max_iou_box[3]), (255, 0, 0), 2)
            #cv2.rectangle(ROI_mask_frame, (ROI_box_last_target[0], ROI_box_last_target[1]), (ROI_box_last_target[2], ROI_box_last_target[3]), (0,255,0), 2)

            # show the frame with mask and two bounding boxes for this precition
            #cv2.imshow("full_mask", full_mask_frame)
            #cv2.imshow(str(frame_name_id) + " : " + str(box_last_id), ROI_mask_frame)
            #cv2.waitKey(0)

            '''
            box_iou_list_copy.pop(max_box_iou_index_t)
            full_box_list_t_copy.pop(max_box_iou_index_t)
            ROI_box_list_t_copy.pop(max_box_iou_index_t)
            full_mask_list_t_copy.pop(max_box_iou_index_t)
            ROI_mask_list_t_copy.pop(max_box_iou_index_t)
            '''

            #cv2.destroyAllWindows()

            #state_t = generate_state(ROI_box_last_target, ROI_correct_last_frame, ROI_max_iou_mask, cropped_frame, w, h, normalization, resnet50)
            #state_t = generate_state(full_box_last_target, correct_last_frame, full_max_iou_mask, frame_numpy, w, h, normalization, resnet50)
            state_t = generate_state_2(ROI_max_iou_mask, cropped_frame, normalization, resnet50)

            dist, value = actor(state_t), critic(state_t)
            action = dist.sample()
            log_prob = dist.log_prob(action).unsqueeze(0)

            # replace the bounding box of last frame
            if action == 0:
              #full_box_list_last[target_last_id] = full_max_iou_box
              #ROI_box_list_last[target_last_id] = ROI_max_iou_box

              full_mask_list_last[target_last_id] = full_max_iou_mask

              correct_last_frame = frame_numpy
              ROI_correct_last_frame = correct_last_frame[top_y:bottom_y, top_x:bottom_x, :]

            #reward_box_iou = solve_coincide(full_box_list_last[target_last_id], label_t_box_full_list[target_last_id])
            #reward_mask_iou = db_eval_iou(full_max_iou_mask, label_t_mask_full_list[target_last_id])

            if target_values[target_last_id] not in pred_values:
              reward_mask_iou = 1.0
            else:
              real_target_value = target_values[target_last_id]
              new_target_id = np.where(pred_values == real_target_value)[0][0]
              reward_mask_iou = db_eval_iou(full_max_iou_mask, label_t_mask_full_list[new_target_id])

            #reward = reward_box_iou
            #reward = reward_mask_iou * reward_box_iou *  reward_box_iou * 100
            reward = reward_mask_iou * reward_mask_iou  * reward_mask_iou * 100

            # save the mask result
            result_mask_path = result_frame_path + "_" + str(target_last_id) + ".png"

            cv2.imwrite(result_mask_path, full_max_iou_mask)

            if frame_name_id == len(frame_name_list)-1:
              done = True
            else:
              done = False

            train_data[target_last_id][0].append(log_prob)
            train_data[target_last_id][1].append(value)
            train_data[target_last_id][2].append(torch.tensor([reward], dtype=torch.float, device=device))
            train_data[target_last_id][3].append(torch.tensor([1 - done], dtype=torch.float, device=device))
            train_data[target_last_id][4].append(reward_mask_iou)
            train_data[target_last_id][5].append(action)
            train_data[target_last_id].append(0.0)

          box_from_mask_list_last = mask_list2box(full_mask_list_last)

          # update the bigger bounding box
          top_x, top_y, bottom_x, bottom_y = generate_ROI(box_from_mask_list_last, original_w, original_h, search_region_ration)

        target_average_acc = 0

        print()
        print("e: "+str(e)+", " + sequence + ", ", end="")

        flog = open("./log", "a")
        flog.write("\n" + "e: "+str(e)+", " + sequence + ", ")
        flog.close()

        for target_last_id in range(len(full_box_list_last)):

          ious = train_data[target_last_id][4]
          ious_mean = sum(ious) / len(ious)
          ious_mean = round(ious_mean, 4)

          log_probs = train_data[target_last_id][0]
          values = train_data[target_last_id][1]
          rewards = train_data[target_last_id][2]
          dones =train_data[target_last_id][3]
          next_value = train_data[target_last_id][6]

          returns = compute_returns(next_value, rewards, dones)
          log_probs = torch.cat(log_probs)
          returns = torch.cat(returns).detach()
          values = torch.cat(values)
          advantage = returns - values

          actor_loss = -(log_probs * advantage.detach()).mean()
          critic_loss = advantage.pow(2).mean()

          if (eval_count%2) == 0:
            optimizerA.zero_grad()
            optimizerC.zero_grad()
            actor_loss.backward(retain_graph=True)
            critic_loss.backward(retain_graph=True)
            optimizerA.step()
            optimizerC.step()
            torch.cuda.empty_cache()

            loss = round(float(actor_loss.data.cpu().numpy()), 4)
            print(str(loss) + ", " + str(ious_mean) + " | ", end="")

            flog = open("./log", "a")
            flog.write(str(loss) + ", " + str(ious_mean) + " | ")
            flog.close()

          target_average_acc += ious_mean

        target_average_acc = target_average_acc/len(full_box_list_last)

        total_accuracy += target_average_acc

        if show_action == True:

          flog = open("./log", "a")

          print()
          flog.write("\n")
          for target_last_id in range(len(full_box_list_last)):
            for a in train_data[target_last_id][5]:
              print(str(a.data.cpu().numpy()), end="")
              flog.write(str(a.data.cpu().numpy()))
            print(", ", end="")
            flog.write(", ")
          print()
          flog.write("\n")

          flog.close()

        if (e % lr_decay_iters == 0) and (e > 0):
          print("learning rate decayed, ", end="")
          for g in optimizerA.param_groups:
            g["lr"] = g["lr"] * 0.99
            print("actor: " + str(g["lr"]) + ", ", end="")

          for g in optimizerC.param_groups:
            g["lr"] = g["lr"] * 0.99
            print("critic: " + str(g["lr"]), end="")


      if (eval_count%2) != 0:

        total_accuracy = total_accuracy / len(sequence_list)
        print()
        print("total accuracy: " + str(total_accuracy))
        print("best_accuracy: " + str(best_accuracy))
        print()
        print()

        flog = open("./log", "a")
        flog.write("\n")
        flog.write("total accuracy: " + str(total_accuracy))
        flog.write("best_accuracy: " + str(best_accuracy))
        flog.write("\n")
        flog.write("\n")
        flog.close()

        f = open("./result", "a")
        f.write(str(total_accuracy) + "\n")
        f.close()

        if total_accuracy > best_accuracy:
          best_accuracy = total_accuracy

          target_path = "./best_results"
          source_path = "./results"
          if not os.path.exists(target_path):
            os.makedirs(target_path)
          if os.path.exists(source_path):
            shutil.rmtree(target_path)
          shutil.copytree(source_path, target_path)
    except:
      print(end="")
  print("done!")

def evalimages(net:Yolact, input_folder:str, output_folder:str):
  if not os.path.exists(output_folder):
    os.mkdir(output_folder)

  print()
  for p in Path(input_folder).glob('*'):
    path = str(p)
    name = os.path.basename(path)
    name = '.'.join(name.split('.')[:-1]) + '.png'
    out_path = os.path.join(output_folder, name)

    evalimage(net, path, out_path)
    print(path + ' -> ' + out_path)
  print('Done.')

from multiprocessing.pool import ThreadPool

def evalvideo(net:Yolact, path:str):
  vid = cv2.VideoCapture(path)
  transform = FastBaseTransform()
  frame_times = MovingAverage()
  fps = 0
  frame_time_target = 1 / vid.get(cv2.CAP_PROP_FPS)

  def cleanup_and_exit():
    print()
    pool.terminate()
    vid.release()
    cv2.destroyAllWindows()
    exit()

  def get_next_frame(vid):
    return [vid.read()[1] for _ in range(args.video_multiframe)]

  def transform_frame(frames):
    with torch.no_grad():
      frames = [torch.Tensor(frame).float().cuda() for frame in frames]
      return frames, transform(torch.stack(frames, 0))

  def eval_network(inp):
    with torch.no_grad():
      frames, imgs = inp
      return frames, net(imgs)

  def prep_frame(inp):
    with torch.no_grad():
      frame, preds = inp
      return prep_display(preds, frame, None, None, None, None, undo_transform=False, class_color=True)

  extract_frame = lambda x, i: (x[0][i], [x[1][i]])

  # Prime the network on the first frame because I do some thread unsafe things otherwise
  print('Initializing model... ', end='')
  eval_network(transform_frame(get_next_frame(vid)))
  print('Done.')

  # For each frame the sequence of functions it needs to go through to be processed (in reversed order)
  sequence = [prep_frame, eval_network, transform_frame]
  pool = ThreadPool(processes=len(sequence) + args.video_multiframe)

  active_frames = []

  print()
  while vid.isOpened():
    start_time = time.time()

    # Start loading the next frames from the disk
    next_frames = pool.apply_async(get_next_frame, args=(vid,))

    # For each frame in our active processing queue, dispatch a job
    # for that frame using the current function in the sequence
    for frame in active_frames:
      frame['value'] = pool.apply_async(sequence[frame['idx']], args=(frame['value'],))

    # For each frame whose job was the last in the sequence (i.e. for all final outputs)
    for frame in active_frames:
      if frame['idx'] == 0:
        # Wait here so that the frame has time to process and so that the video plays at the proper speed
        time.sleep(frame_time_target)

        cv2.imshow(path, frame['value'].get())
        if cv2.waitKey(1) == 27: # Press Escape to close
          cleanup_and_exit()

    # Remove the finished frames from the processing queue
    active_frames = [x for x in active_frames if x['idx'] > 0]

    # Finish evaluating every frame in the processing queue and advanced their position in the sequence
    for frame in list(reversed(active_frames)):
      frame['value'] = frame['value'].get()
      frame['idx'] -= 1

      if frame['idx'] == 0:
        # Split this up into individual threads for prep_frame since it doesn't support batch size
        active_frames += [{'value': extract_frame(frame['value'], i), 'idx': 0} for i in range(1, args.video_multiframe)]
        frame['value'] = extract_frame(frame['value'], 0)

    # Finish loading in the next frames and add them to the processing queue
    active_frames.append({'value': next_frames.get(), 'idx': len(sequence)-1})

    # Compute FPS
    frame_times.add(time.time() - start_time)
    fps = args.video_multiframe / frame_times.get_avg()

    print('\rAvg FPS: %.2f     ' % fps, end='')

  cleanup_and_exit()

def savevideo(net:Yolact, in_path:str, out_path:str):

  vid = cv2.VideoCapture(in_path)

  target_fps   = round(vid.get(cv2.CAP_PROP_FPS))
  frame_width  = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
  num_frames   = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))

  out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height))

  transform = FastBaseTransform()
  frame_times = MovingAverage()
  progress_bar = ProgressBar(30, num_frames)

  try:
    for i in range(num_frames):
      timer.reset()
      with timer.env('Video'):
        frame = torch.Tensor(vid.read()[1]).float().cuda()
        batch = transform(frame.unsqueeze(0))
        preds = net(batch)
        processed = prep_display(preds, frame, None, None, None, None, undo_transform=False, class_color=True)

        out.write(processed)

      if i > 1:
        frame_times.add(timer.total_time())
        fps = 1 / frame_times.get_avg()
        progress = (i+1) / num_frames * 100
        progress_bar.set_val(i+1)

        print('\rProcessing Frames  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
              % (repr(progress_bar), i+1, num_frames, progress, fps), end='')
  except KeyboardInterrupt:
    print('Stopping early.')

  vid.release()
  out.release()
  print()


def evaluate(net:Yolact, dataset, train_mode=False):
  net.detect.cross_class_nms = args.cross_class_nms
  net.detect.use_fast_nms = args.fast_nms
  cfg.mask_proto_debug = args.mask_proto_debug

  if args.image is not None:
    if ':' in args.image:
      inp, out = args.image.split(':')
      evalimage(net, inp, out)
    else:
      evalimage(net, args.image)
    return

  elif args.images is not None:
    inp, out = args.images.split(':')
    evalimages(net, inp, out)
    return

  elif args.video is not None:
    if ':' in args.video:
      inp, out = args.video.split(':')
      savevideo(net, inp, out)
    else:
      evalvideo(net, args.video)
    return

  frame_times = MovingAverage()
  dataset_size = len(dataset) if args.max_images < 0 else min(args.max_images, len(dataset))
  progress_bar = ProgressBar(30, dataset_size)

  print()

  if not args.display and not args.benchmark:
    # For each class and iou, stores tuples (score, isPositive)
    # Index ap_data[type][iouIdx][classIdx]
    ap_data = {
      'box' : [[APDataObject() for _ in COCO_CLASSES] for _ in iou_thresholds],
      'mask': [[APDataObject() for _ in COCO_CLASSES] for _ in iou_thresholds]
    }
    detections = Detections()
  else:
    timer.disable('Load Data')

  dataset_indices = list(range(len(dataset)))

  if args.shuffle:
    random.shuffle(dataset_indices)
  elif not args.no_sort:
    # Do a deterministic shuffle based on the image ids
    #
    # I do this because on python 3.5 dictionary key order is *random*, while in 3.6 it's
    # the order of insertion. That means on python 3.6, the images come in the order they are in
    # in the annotations file. For some reason, the first images in the annotations file are
    # the hardest. To combat this, I use a hard-coded hash function based on the image ids
    # to shuffle the indices we use. That way, no matter what python version or how pycocotools
    # handles the data, we get the same result every time.
    hashed = [badhash(x) for x in dataset.ids]
    dataset_indices.sort(key=lambda x: hashed[x])

  dataset_indices = dataset_indices[:dataset_size]

  try:
    # Main eval loop
    for it, image_idx in enumerate(dataset_indices):
      timer.reset()

      with timer.env('Load Data'):
        img, gt, gt_masks, h, w, num_crowd = dataset.pull_item(image_idx)

        # Test flag, do not upvote
        if cfg.mask_proto_debug:
          with open('scripts/info.txt', 'w') as f:
            f.write(str(dataset.ids[image_idx]))
          np.save('scripts/gt.npy', gt_masks)

        batch = Variable(img.unsqueeze(0))
        if args.cuda:
          batch = batch.cuda()

      with timer.env('Network Extra'):
        preds = net(batch)

      # Perform the meat of the operation here depending on our mode.
      if args.display:
        img_numpy = prep_display(preds, img, gt, gt_masks, h, w)
      elif args.benchmark:
        prep_benchmark(preds, h, w)
      else:
        prep_metrics(ap_data, preds, img, gt, gt_masks, h, w, num_crowd, dataset.ids[image_idx], detections)

      # First couple of images take longer because we're constructing the graph.
      # Since that's technically initialization, don't include those in the FPS calculations.
      if it > 1:
        frame_times.add(timer.total_time())

      if args.display:
        if it > 1:
          print('Avg FPS: %.4f' % (1 / frame_times.get_avg()))
        plt.imshow(img_numpy)
        plt.title(str(dataset.ids[image_idx]))
        plt.show()
      elif not args.no_bar:
        if it > 1: fps = 1 / frame_times.get_avg()
        else: fps = 0
        progress = (it+1) / dataset_size * 100
        progress_bar.set_val(it+1)
        print('\rProcessing Images  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
              % (repr(progress_bar), it+1, dataset_size, progress, fps), end='')

    if not args.display and not args.benchmark:
      print()
      if args.output_coco_json:
        print('Dumping detections...')
        if args.output_web_json:
          detections.dump_web()
        else:
          detections.dump()
      else:
        if not train_mode:
          print('Saving data...')
          with open(args.ap_data_file, 'wb') as f:
            pickle.dump(ap_data, f)

        return calc_map(ap_data)
    elif args.benchmark:
      print()
      print()
      print('Stats for the last frame:')
      timer.print_stats()
      avg_seconds = frame_times.get_avg()
      print('Average: %5.2f fps, %5.2f ms' % (1 / frame_times.get_avg(), 1000*avg_seconds))

  except KeyboardInterrupt:
    print('Stopping...')


def calc_map(ap_data):
  print('Calculating mAP...')
  aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

  for _class in range(len(COCO_CLASSES)):
    for iou_idx in range(len(iou_thresholds)):
      for iou_type in ('box', 'mask'):
        ap_obj = ap_data[iou_type][iou_idx][_class]

        if not ap_obj.is_empty():
          aps[iou_idx][iou_type].append(ap_obj.get_ap())

  all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

  # Looking back at it, this code is really hard to read :/
  for iou_type in ('box', 'mask'):
    all_maps[iou_type]['all'] = 0 # Make this first in the ordereddict
    for i, threshold in enumerate(iou_thresholds):
      mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
      all_maps[iou_type][int(threshold*100)] = mAP
    all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values())-1))

  print_maps(all_maps)
  return all_maps

def print_maps(all_maps):
  # Warning: hacky
  make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
  make_sep = lambda n:  ('-------+' * n)

  print()
  print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
  print(make_sep(len(all_maps['box']) + 1))
  for iou_type in ('box', 'mask'):
    print(make_row([iou_type] + ['%.2f' % x for x in all_maps[iou_type].values()]))
  print(make_sep(len(all_maps['box']) + 1))
  print()



if __name__ == '__main__':
  parse_args()

  if args.config is not None:
    set_cfg(args.config)

  if args.trained_model == 'interrupt':
    args.trained_model = SavePath.get_interrupt('weights/')
  elif args.trained_model == 'latest':
    args.trained_model = SavePath.get_latest('weights/', cfg.name)

  if args.config is None:
    model_path = SavePath.from_str(args.trained_model)
    # TODO: Bad practice? Probably want to do a name lookup instead.
    args.config = model_path.model_name + '_config'
    print('Config not specified. Parsed %s from the file name.\n' % args.config)
    set_cfg(args.config)

  if args.detect:
    cfg.eval_mask_branch = False

  if args.dataset is not None:
    set_dataset(args.dataset)

  with torch.no_grad():
    if not os.path.exists('results'):
      os.makedirs('results')

    if args.cuda:
      cudnn.benchmark = True
      cudnn.fastest = True
      torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
      torch.set_default_tensor_type('torch.FloatTensor')

    if args.resume and not args.display:
      with open(args.ap_data_file, 'rb') as f:
        ap_data = pickle.load(f)
      calc_map(ap_data)
      exit()

    if args.image is None and args.video is None and args.images is None:
      dataset = COCODetection(cfg.dataset.valid_images, cfg.dataset.valid_info, transform=BaseTransform())
      prep_coco_cats(dataset.coco.cats)
    else:
      dataset = None

    print('Loading model...', end='')
    net = Yolact()
    net.load_weights(args.trained_model)
    net.eval()
    print(' Done.')

    if args.cuda:
      net = net.cuda()

  evaluate(net, dataset)
