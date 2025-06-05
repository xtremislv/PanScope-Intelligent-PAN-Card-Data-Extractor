import cv2
import matplotlib.pyplot as plt
import numpy as np
import re
from doctr.file_utils import is_tf_available
from doctr.io import DocumentFile
from doctr.utils.visualization import visualize_page
import json
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
file_path = file_path = os.path.join(ROOT_DIR, 'Pan Cards.v1i.yolov12', 'train', 'images', '46_jpg.rf.f4ce07b031e538978d76cecf3d7a7d73.jpg')
os.environ["USE_TORCH"] = "1"
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

if file_path.lower().endswith(".pdf"):
    with open(file_path, "rb") as f:
        doc = DocumentFile.from_pdf(f.read())
else:
    doc = DocumentFile.from_images(file_path)

page = doc[0]

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

out = predictor([page])
page_export = out.pages[0].export()

op = json.dumps(page_export, indent=2)
with open('ocr_output.txt', 'w') as f:
    f.write(op)



def remove_fields(obj, fields):
    if isinstance(obj, list):
        for item in obj:
            remove_fields(item, fields)
    elif isinstance(obj, dict):
        for key in list(obj.keys()):
            if key in fields:
                del obj[key]
            else:
                remove_fields(obj[key], fields)

def remove_geometry(data):
    if isinstance(data, list):
        for item in data:
            remove_geometry(item)
    elif isinstance(data, dict):
        if 'geometry' in data:
            del data['geometry']
        for key, value in data.items():
            remove_geometry(value)

def extract_pan_and_dob_positions(blocks):
    pan_pattern = r"[A-Z]{5}[0-9]{4}[A-Z]"
    dob_pattern = r"\d{2}/\d{2}/\d{4}"

    pan_info = None
    dob_info = None

    for block in blocks:
        for line in block.get("lines", []):
            for word in line.get("words", []):
                value = word.get("value", "").strip()
                geometry = word.get("geometry", [])
                if not geometry or len(geometry) != 2:
                    continue
                y_min = geometry[0][1]

                if re.fullmatch(pan_pattern, value) and not pan_info:
                    pan_info = (value, y_min)

                if re.fullmatch(dob_pattern, value) and not dob_info:
                    dob_info = (value, y_min)

                if pan_info and dob_info:
                    return pan_info, dob_info

    return pan_info, dob_info

def detect_layout_by_relative_position(blocks):
    pan_info, dob_info = extract_pan_and_dob_positions(blocks)
    if pan_info and dob_info:
        pan_y = pan_info[1]
        dob_y = dob_info[1]
        if pan_y > dob_y:
            return "old", pan_info[0]
        else:
            return "new", pan_info[0]
    return "unknown", None

def extract_pan_info(ocr_json):
    if isinstance(ocr_json, str):
        import json
        ocr_json = json.loads(ocr_json)

    blocks = ocr_json.get("blocks", [])
    layout, pan = detect_layout_by_relative_position(blocks)

    if layout == "old":
        return extract_info_old(ocr_json)
    elif layout == "new":
        return extract_info_new(ocr_json)
    else:
        return {"error": "Unable to determine layout", "pan": pan}


def extract_info_old(ocr_json):
    lines = []
    # Step 1: Collect all text lines
    for block in ocr_json.get("blocks", []):
        for line in block.get("lines", []):
            line_text = " ".join(word.get("value", "") for word in line.get("words", []))
            lines.append(line_text)

    # Step 2: Clean and filter lines
    lines = [line.strip() for line in lines if line.strip()]
    
    # Step 3: Find DOB and PAN using regex
    dob_pattern = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
    pan_pattern = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")

    dob = next((line for line in lines if dob_pattern.search(line)), None)
    pan = next((line for line in lines if pan_pattern.search(line)), None)

    # Step 4: Get indices to extract name and father's name
    dob_index = lines.index(dob) if dob else -1
    pan_index = lines.index(pan) if pan else -1

    name = lines[dob_index - 2] if dob_index >= 2 else None
    father_name = lines[dob_index - 1] if dob_index >= 1 else None

    return {
        "layout": "old",
        "Name": name,
        "Father's Name": father_name,
        "Date of Birth": dob,
        "PAN Number": pan
    }

def extract_info_new(ocr_json):
    lines = []
    # Step 1: Collect all text lines
    for block in ocr_json.get("blocks", []):
        for line in block.get("lines", []):
            line_text = " ".join(word.get("value", "") for word in line.get("words", []))
            lines.append(line_text.strip())

    # Step 2: Clean and filter lines
    lines = [line for line in lines if line]

    # Step 3: Find DOB and PAN using regex
    dob_pattern = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
    pan_pattern = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")

    dob = next((line for line in lines if dob_pattern.search(line)), None)
    pan = next((line for line in lines if pan_pattern.search(line)), None)

    # Step 4: Find name and father's name based on keyword context
    name = None
    father_name = None

    for i, line in enumerate(lines):
        # Match the name heading only if it is NOT part of "Father's Name"
        if "name" in line.lower() and "father" not in line.lower() and i + 1 < len(lines):
           name_candidate = lines[i + 1].strip()
           if name_candidate and all(x not in name_candidate.lower() for x in ["name", "father"]):
               name = name_candidate

        # Match father's name specifically
        if "father" in line.lower() and i + 1 < len(lines):
           father_candidate = lines[i + 1].strip()
           if father_candidate and "father" not in father_candidate.lower():
               father_name = father_candidate

    return {
        "layout": "new",
        "Name": name,
        "Father's Name": father_name,
        "Date of Birth": dob,
        "PAN Number": pan
    }


def is_possible_name(line):
    line = line.strip()
    if not line or len(line) < 2:
        return False
    if line.lower() in ["-", "signature", "date of birth", "permanent account number", "income tax"]:
        return False
    if not re.search(r"[A-Za-z]", line):  # Must contain letters
        return False
    if len(line) <= 2 and line.isupper():
        return False
    return True

def extract_full_text(blocks):
    lines = []
    for block in blocks:
        for line in block.get("lines", []):
            line_text = " ".join([word.get("value", "") for word in line.get("words", [])])
            lines.append(line_text.strip())
    return "\n".join(lines)


result = extract_pan_info(op)
print(result)


