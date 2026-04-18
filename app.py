import spacy
import os
import torch
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import traceback
import numpy as np
import urllib.request
from ultralytics import YOLO
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel
)

MODEL_NAME = "en_core_web_sm"

# load nlp model (download if missing)
try:
    nlp = spacy.load(MODEL_NAME)
except OSError:
    print(f"Downloading {MODEL_NAME}...")
    from spacy.cli import download
    download(MODEL_NAME)
    nlp = spacy.load(MODEL_NAME)

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Launching on device: {device}")
print("Loading Models...")

# BLIP
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# YOLO-World
yolo_model = YOLO('yolov8s-world.pt').to(device)

# CLIP
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=model_dtype).to(device)

# extract nouns from BLIP caption
def extract_nouns(caption: str):
    doc = nlp(caption.lower())
    ignore_list = ["image", "photo", "picture", "background", "foreground"]
    keywords = [token.text for token in doc 
                if token.pos_ in ["NOUN", "PROPN", "PRON"] 
                and token.text not in ignore_list]
    return list(set(keywords))

# process image using models
@torch.no_grad()
def process_image(image: Image.Image):
    # captioning
    prompt = "a photo of"
    blip_inputs = blip_processor(image, text=prompt, return_tensors="pt").to(device)
    
    # cast to half-precision if using GPU
    if device.type == "cuda":
        blip_inputs["pixel_values"] = blip_inputs["pixel_values"].to(torch.float16)
        
    generated_ids = blip_model.generate(
        **blip_inputs,
        max_new_tokens=30,
        num_beams=5,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True).title()

    # object detection (YOLO-World)
    queries = extract_nouns(caption)
    if not queries:
        queries = ["object"]

    yolo_model.set_classes(queries)
    results = yolo_model.predict(source=image, conf=0.15, iou=0.45, verbose=False)[0]

    raw_detections = []
    for box in results.boxes:
        label_name = yolo_model.names[int(box.cls[0])].title()
        confidence = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        raw_detections.append({
            "label": label_name,
            "confidence": round(confidence * 100, 1),
            "box": [round(i, 2) for i in coords]
        })

    # fallback if nothing found
    if len(raw_detections) == 0:
        fallback_classes = ["person", "vehicle", "animal", "furniture", "building", "structure", "bag", "object"]
        yolo_model.set_classes(fallback_classes)
        fallback_results = yolo_model.predict(source=image, conf=0.20, iou=0.45, verbose=False)
        for result in fallback_results:
            for box in result.boxes:
                raw_detections.append({
                    "label": yolo_model.names[int(box.cls[0])].title(),
                    "confidence": round(float(box.conf[0]) * 100, 1),
                    "box": [round(i, 2) for i in box.xyxy[0].tolist()]
                })

    detected_items = sorted(raw_detections, key=lambda x: x["confidence"], reverse=True)[:15]

    # scene classification (CLIP)
    scene_categories = {
        "Indoor Space": "a photo with an indoor room background setting",
        "Nature or Wildlife": "a photo with an outdoor natural background",
        "Urban City": "a photo with an outdoor urban city background",
        "City Skyline": "a photo showing a distant city skyline",
        "Crowded Area": "a photo showing a dense crowd of people"
    }
    ui_labels = list(scene_categories.keys())
    clip_prompts = list(scene_categories.values())

    clip_inputs = clip_processor(text=clip_prompts, images=image, return_tensors="pt", padding=True).to(device)
    if device.type == "cuda":
        clip_inputs["pixel_values"] = clip_inputs["pixel_values"].to(torch.float16)
        
    clip_outputs = clip_model(**clip_inputs)
    probs = clip_outputs.logits_per_image.softmax(dim=1).squeeze().tolist()

    scene_results = sorted(
        [{"label": label, "probability": round(p * 100, 1)} for label, p in zip(ui_labels, probs)],
        key=lambda x: x["probability"],
        reverse=True
    )
    return caption, detected_items, scene_results

# download font if missing
font_path = "Roboto-Bold.ttf"
if not os.path.exists(font_path):
    urllib.request.urlretrieve("https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Bold.ttf", font_path)


def inference_wrapper(image):
    if image is None:
        return None, "ERROR: No image detected.", "Waiting for image...", {"Error": 1.0}
    try:
        process_img = image.convert("RGB")

        # run pipeline
        caption, detected_items, scene_results = process_image(process_img)
    
        # create copy for drawing bounding boxes
        annotated_image = process_img.copy()
        draw = ImageDraw.Draw(annotated_image)
        text_size = max(20, int(annotated_image.width * 0.03))
        font = ImageFont.truetype(font_path, text_size)

        yolo_labels_clean = []

        # draw detections
        for item in detected_items:
            box = item["box"]

            # format label text with confidence score
            label_text = f"{item['label']} ({item['confidence']}%)"
            yolo_labels_clean.append(label_text)

            box_coords = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]

            # draw bounding box
            line_width = max(3, int(annotated_image.width * 0.006))
            draw.rectangle(box_coords, outline="red", width=line_width)

            # position label above bounding box
            text_x = box_coords[0]
            text_y = max(0, box_coords[1] - (text_size + int(text_size * 0.2)))

            text_bbox = draw.textbbox((text_x, text_y), label_text, font=font)
            padding = int(text_size * 0.2)
            padded_bbox = [text_bbox[0]-padding, text_bbox[1]-padding, text_bbox[2]+padding, text_bbox[3]+padding]
            
            # draw label background + text
            draw.rectangle(padded_bbox, fill="red")
            draw.text((text_x, text_y), label_text, fill="white", font=font)

        yolo_text_output = ", ".join(yolo_labels_clean) if yolo_labels_clean else "No identifiable objects detected." # format detection output 
        clip_chart_data = {item["label"]: float(item["probability"]) / 100.0 for item in scene_results}               # convert classification output into chart

        return np.array(annotated_image), caption, yolo_text_output, clip_chart_data
    
    except Exception as e:
        print(traceback.format_exc())
        return np.array(image), f"SYSTEM CRASH: {str(e)}", "Error", {"Error": 1.0}

# UI
with gr.Blocks() as demo:
    gr.Markdown("<div style='text-align: center;'><h1>Clarity: Semantic Vision Engine</h1><p>Dynamic Multi-Modal Pipeline</p></div>")
    with gr.Row():
        with gr.Column():
            shared_image = gr.Image(type="pil", label="Input/Output", height=550)
            submit_btn = gr.Button("Analyze Image", variant="primary")
        with gr.Column():
            gr.Markdown("<h3 style='text-align: center;'>Scene Intelligence</h3>")
            out_caption = gr.Textbox(label="1. SEMANTIC NARRATIVE (BLIP)", lines=2, interactive=False)
            out_scene = gr.Label(label="2. SCENE CONTEXT (ZERO-SHOT CLIP)", num_top_classes=3)
            out_yolo_text = gr.Textbox(label="3. DETECTED OBJECTS (YOLO-WORLD)", lines=2, interactive=False)

    submit_btn.click(fn=inference_wrapper, inputs=shared_image, outputs=[shared_image, out_caption, out_yolo_text, out_scene])


demo.launch()
