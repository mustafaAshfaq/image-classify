from collections import Counter
from functools import lru_cache

import gradio as gr
from PIL import Image, ImageDraw, ImageOps


CAPTION_MODEL_ID = "Salesforce/blip-image-captioning-base"
DETECTOR_ID = "facebook/detr-resnet-50"


@lru_cache(maxsize=1)
def get_captioner():
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(CAPTION_MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_ID).to(device)
    model.eval()
    return processor, model, device


@lru_cache(maxsize=1)
def get_detector():
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection, DetrConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(DETECTOR_ID, revision="no_timm")
    config_dict, _ = DetrConfig.get_config_dict(DETECTOR_ID, revision="no_timm")
    if config_dict.get("dilation") is None:
        config_dict["dilation"] = False
    config = DetrConfig.from_dict(config_dict)
    model = AutoModelForObjectDetection.from_pretrained(
        DETECTOR_ID, revision="no_timm", config=config
    ).to(device)
    model.eval()
    return processor, model, device


def format_detections(image, detections, labels):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    rows = []
    counts = Counter()
    width, height = image.size
    for score, label_id, box in zip(
        detections["scores"], detections["labels"], detections["boxes"]
    ):
        label = labels[int(label_id)]
        counts[label] += 1
        left, top, right, bottom = box.tolist()
        left, right = sorted((max(0, min(width - 1, left)), max(0, min(width - 1, right))))
        top, bottom = sorted((max(0, min(height - 1, top)), max(0, min(height - 1, bottom))))
        center_x = (left + right) / (2 * width)
        center_y = (top + bottom) / (2 * height)
        horizontal = "left" if center_x < 1 / 3 else "right" if center_x > 2 / 3 else "center"
        vertical = "top" if center_y < 1 / 3 else "bottom" if center_y > 2 / 3 else "middle"
        coverage = (right - left) * (bottom - top) / (width * height) * 100
        confidence = round(float(score) * 100, 1)
        name = f"{label} {counts[label]}"
        rows.append([name, confidence, f"{vertical}-{horizontal}", round(coverage, 1)])
        draw.rectangle((left, top, right, bottom), outline="#00a884", width=3)
        text = f"{name} ({confidence}%)"
        text_box = draw.textbbox((0, 0), text)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_x = max(0, min(left, width - text_width - 8))
        text_y = max(0, top - text_height - 8)
        draw.rectangle(
            (text_x, text_y, text_x + text_width + 8, text_y + text_height + 8),
            fill="#00a884",
        )
        draw.text((text_x + 4, text_y + 4 - text_box[1]), text, fill="white")

    inventory = "\n".join(f"- **{label}**: {count}" for label, count in sorted(counts.items()))
    report = f"### Objects found: {len(rows)}\n\n{inventory}" if rows else "### No objects detected at this confidence level"
    report += (
        "\n\nDetection covers 80 common object categories, not every possible item. "
        "Small or obscured objects may be missed; counts and captions are model estimates."
    )
    return annotated, report, rows


def describe_image(image: Image.Image | None, confidence: float = 0.7):
    import torch

    if image is None:
        raise gr.Error("Upload an image to analyze.")

    image = ImageOps.exif_transpose(image).convert("RGB")
    processor, model, device = get_captioner()
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=80)
    caption = processor.decode(generated_ids[0], skip_special_tokens=True).strip()

    detector_processor, detector, device = get_detector()
    inputs = detector_processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = detector(**inputs)
    detections = detector_processor.post_process_object_detection(
        outputs, threshold=confidence, target_sizes=[(image.height, image.width)]
    )[0]
    annotated, report, rows = format_detections(image, detections, detector.config.id2label)
    return annotated, report, rows, caption


with gr.Blocks(title="Image Details") as app:
    gr.Markdown("# Image Details")
    with gr.Row():
        image_input = gr.Image(label="Image", type="pil")
        annotated_output = gr.Image(label="Detected objects", type="pil", interactive=False)
    confidence_input = gr.Slider(
        minimum=0.1, maximum=0.95, value=0.7, step=0.05, label="Minimum confidence"
    )
    describe_button = gr.Button("Analyze image", variant="primary")
    description_output = gr.Textbox(label="Scene overview", lines=2, interactive=False)
    inventory_output = gr.Markdown()
    details_output = gr.Dataframe(
        headers=["Object", "Confidence (%)", "Position", "Image area (%)"],
        datatype=["str", "number", "str", "number"],
        label="Object details",
        interactive=False,
    )
    describe_button.click(
        fn=describe_image,
        inputs=[image_input, confidence_input],
        outputs=[annotated_output, inventory_output, details_output, description_output],
        concurrency_limit=1,
    )


if __name__ == "__main__":
    app.launch()