import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from mm import extract_pan_info
import json
from doctr.file_utils import is_tf_available
from doctr.io import DocumentFile
from doctr.utils.visualization import visualize_page
os.environ["USE_TORCH"] = "1"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

if is_tf_available():
    import tensorflow as tf
    from demo.backend.tensorflow import DET_ARCHS, RECO_ARCHS, forward_image, load_predictor

    if any(tf.config.experimental.list_physical_devices("gpu")):
        forward_device = tf.device("/gpu:0")
    else:
        forward_device = tf.device("/cpu:0")

else:
    import torch
    from demo.backend.pytorch import DET_ARCHS, RECO_ARCHS, forward_image, load_predictor

    forward_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

det_arch = DET_ARCHS[0]
reco_arch = RECO_ARCHS[0]

predictor = load_predictor(
    det_arch=det_arch,
    reco_arch=reco_arch,
    assume_straight_pages=True,
    straighten_pages=False,
    export_as_straight_boxes=True,
    disable_page_orientation=False,
    disable_crop_orientation=True,
    bin_thresh=0.3,
    box_thresh=0.1,
    device=forward_device,
)
import os
import cv2
import glob
import numpy as np
from sklearn.metrics import roc_auc_score

# ---- Utility: Convert YOLO bbox to (x1, y1, x2, y2)
def yolo_to_box(label, img_width, img_height):
    class_id, x_center, y_center, w, h = map(float, label.strip().split())
    x_center *= img_width
    y_center *= img_height
    w *= img_width
    h *= img_height
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    return [x1, y1, x2, y2]

# ---- Utility: IoU Calculation
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1e-6, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return interArea / (boxAArea + boxBArea - interArea + 1e-6)


# ---- Utility: Extract boxes from heatmap
def extract_pred_boxes_from_heatmap(seg_map, threshold=0.3):
    boxes = []
    _, bin_map = cv2.threshold(seg_map, threshold, 1, cv2.THRESH_BINARY)
    bin_map = (bin_map * 255).astype(np.uint8)
    contours, _ = cv2.findContours(bin_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        boxes.append([x, y, x+w, y+h])
    return boxes


import matplotlib.pyplot as plt

def draw_boxes(img, boxes, color='r', has_class=True):
    for box in boxes:
        if has_class:
            if len(box) < 5:
                print(f"Skipping malformed box (expected 5 values): {box}")
                continue
            _, x_center, y_center, w, h = box
        else:
            if len(box) < 4:
                print(f"Skipping malformed box (expected 4 values): {box}")
                continue
            x_center, y_center, w, h = box

        x0 = (x_center - w / 2) * img.shape[1]
        y0 = (y_center - h / 2) * img.shape[0]
        x1 = (x_center + w / 2) * img.shape[1]
        y1 = (y_center + h / 2) * img.shape[0]
        plt.gca().add_patch(
            plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor=color, fill=False, lw=2)
        )

def remove_symmetric_padding_and_resize(seg_map, original_size):
    orig_h, orig_w = original_size
    padded_h, padded_w = seg_map.shape[:2]
    
    scale = min(padded_w / orig_w, padded_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    
    pad_x = (padded_w - new_w) // 2
    pad_y = (padded_h - new_h) // 2
    
    cropped = seg_map[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    
    resized = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return resized

def convert_all_gt_yolo_to_xyxy(gt_yolo_boxes, img_w, img_h):
    converted = []
    for box in gt_yolo_boxes:
        cls, cx, cy, w, h = box
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        converted.append([int(cls), x1, y1, x2, y2])
    return converted

def load_yolo_boxes(label_path):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Malformed line in label file: {line.strip()}")
                continue
            boxes.append([int(parts[0])] + list(map(float, parts[1:])))
    return boxes

def convert_abs_to_normalized(pred_boxes, img_width, img_height):
    normalized_preds = []
    for box in pred_boxes:
        x_min, y_min, x_max, y_max = box
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        normalized_preds.append([x_center, y_center, width, height])
    return normalized_preds

def match_boxes_greedy(gt_boxes, pred_boxes, iou_threshold=0.2):
    matches = []
    used_pred = set()
    used_gt = set()
    
    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))

    for i, gt in enumerate(gt_boxes):
        gt_coords = gt[1:]  # skip class label
        for j, pred in enumerate(pred_boxes):
            iou_matrix[i, j] = iou(gt_coords, pred)

    while True:
        max_iou = iou_matrix.max()
        if max_iou < iou_threshold:
            break
        i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        matches.append((gt_boxes[i], pred_boxes[j], max_iou))
        used_gt.add(i)
        used_pred.add(j)
        iou_matrix[i, :] = -1
        iou_matrix[:, j] = -1

    tp = len(matches)
    fp = len(pred_boxes) - len(used_pred)
    fn = len(gt_boxes) - len(used_gt)
    
    return matches, tp, fp, fn

# ---- Main Evaluation Function
def evaluate_dataset(image_dir=os.path.join(ROOT_DIR ,'Pan Cards.v1i.yolov12', 'train', 'images'), label_dir=os.path.join(ROOT_DIR, 'Pan Cards.v1i.yolov12', 'train', 'labels'), predictor = predictor, device=forward_device):
    all_scores = []
    for img_path in glob.glob(os.path.join(image_dir, "*.jpg")):
        name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f"{name}.txt")


        image = cv2.imread(img_path)
        h, w = image.shape[:2]

        # Load GT boxes
        gt_boxes = load_yolo_boxes(label_path)

        # Predict heatmap
        doc = DocumentFile.from_images(img_path)
        page = doc[0]
        out = predictor([page])
        page_export = out.pages[0].export()
        op = json.dumps(page_export, indent=2)
        result = extract_pan_info(op)
        with open(os.path.join(ROOT_DIR, 'Pan Cards.v1i.yolov12', 'train', 'result',f'{name}.txt'), 'w') as f:
            f.write(json.dumps(result))
        seg_map = forward_image(predictor, page, device)
        seg_map = np.squeeze(seg_map)
        seg_map_fixed = remove_symmetric_padding_and_resize(seg_map, (h, w))

        pred_boxes = extract_pred_boxes_from_heatmap(seg_map_fixed)
        pred_boxes1 = convert_abs_to_normalized(pred_boxes , w , h)
        gt_boxes1 = convert_all_gt_yolo_to_xyxy(gt_boxes,img_w=w, img_h=h)
        matches, tp, fp, fn = match_boxes_greedy(gt_boxes1, pred_boxes, iou_threshold=0.02)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        mean_iou = np.mean([m[2] for m in matches]) if matches else 0
        acc = tp / (tp + fp + fn + 1e-6)

        print(f"{name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Acc={acc:.3f} , Mean_iou={mean_iou:.3f}")
        all_scores.append((precision, recall, f1, acc, mean_iou))

    # Avg metrics
    if all_scores:
        mean_scores = np.mean(all_scores, axis=0)
        print(f"\nAverage across dataset:")
        print(f"Precision: {mean_scores[0]:.3f}")
        print(f"Recall: {mean_scores[1]:.3f}")
        print(f"F1 Score: {mean_scores[2]:.3f}")
        print(f"Accuracy: {mean_scores[3]:.3f}")
        print(f"Mean_iou: {mean_scores[4]:.3f}")

evaluate_dataset(image_dir=os.path.join(ROOT_DIR,'Pan Cards.v1i.yolov12', 'train', 'images'), label_dir=os.path.join(ROOT_DIR, 'Pan Cards.v1i.yolov12', 'train', 'labels'), predictor = predictor, device=forward_device)