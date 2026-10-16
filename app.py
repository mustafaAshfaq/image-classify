from functools import lru_cache

import gradio as gr
from PIL import Image


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


def describe_image(image: Image.Image | None) -> str:
    import torch

    if image is None:
        raise gr.Error("Upload an image to generate a description.")

    processor, model, device = get_captioner()
    inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=80)
    return processor.decode(generated_ids[0], skip_special_tokens=True).strip()


with gr.Blocks(title="Image Description") as app:
    gr.Markdown("# Image Description")
    image_input = gr.Image(label="Image", type="pil")
    describe_button = gr.Button("Describe image", variant="primary")
    description_output = gr.Textbox(label="Description", lines=5, interactive=False)
    describe_button.click(
        fn=describe_image,
        inputs=image_input,
        outputs=description_output,
    )


if __name__ == "__main__":
    app.launch()