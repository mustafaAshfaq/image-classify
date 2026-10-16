import base64
import binascii
from functools import lru_cache
from io import BytesIO

import gradio as gr
from PIL import Image, UnidentifiedImageError


MODEL_ID = "Salesforce/blip-image-captioning-base"


@lru_cache(maxsize=1)
def get_captioner():
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return processor, model, device


def load_image_from_data_url(data_url: str) -> Image.Image:
    data_url = data_url.strip()
    if not data_url.startswith("data:image/") or "," not in data_url:
        raise gr.Error("Provide a valid base64-encoded image data URL.")

    header, encoded = data_url.split(",", 1)
    if ";base64" not in header.lower():
        raise gr.Error("Provide a valid base64-encoded image data URL.")

    try:
        image_bytes = base64.b64decode(encoded.strip(), validate=True)
        with Image.open(BytesIO(image_bytes)) as img:
            return img.copy()
    except (binascii.Error, UnidentifiedImageError, OSError) as exc:
        raise gr.Error("Provide a valid base64-encoded image data URL.") from exc


def describe_image(
    image: Image.Image | str | None = None,
    data_url: str | None = None,
) -> str:
    import torch

    if isinstance(image, str):
        image = load_image_from_data_url(image) if image.strip() else None
    elif image is None and data_url and data_url.strip():
        image = load_image_from_data_url(data_url)

    if image is None:
        raise gr.Error("Upload an image or provide an image data URL to generate a description.")

    processor, model, device = get_captioner()
    inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=80)
    return processor.decode(generated_ids[0], skip_special_tokens=True).strip()


with gr.Blocks(title="Image Description") as app:
    gr.Markdown("# Image Description")
    image_input = gr.Image(label="Image", type="pil")
    data_url_input = gr.Textbox(
        label="Image Data URL",
        placeholder="data:image/png;base64,...",
    )
    describe_button = gr.Button("Describe image", variant="primary")
    description_output = gr.Textbox(label="Description", lines=5, interactive=False)
    describe_button.click(
        fn=describe_image,
        inputs=[image_input, data_url_input],
        outputs=description_output,
    )


if __name__ == "__main__":
    app.launch()