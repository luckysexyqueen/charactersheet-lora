import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import spaces  # MUST come before torch / diffusers / transformers
import torch
import gradio as gr
import numpy as np
import random
from PIL import Image
from diffusers import Flux2KleinPipeline

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"  # distilled, 4-step base
LORA_REPO = "Alissonerdx/CharacterSheet"
LORA_WEIGHT = "QuadView_klein9b_v1.safetensors"  # FLUX.2 Klein 9B QuadView LoRA

TRIGGER_PROMPT = (
    "Convert the character in the image to a Character Sheet showing a face "
    "close-up, front, side and back full body views"
)

dtype = torch.bfloat16
device = "cuda"

print("Loading FLUX.2 Klein 9B pipeline...")
pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype).to("cuda")
print("Loading CharacterSheet QuadView LoRA...")
pipe.load_lora_weights(LORA_REPO, weight_name=LORA_WEIGHT)
print("LoRA loaded. Ready.")

MAX_SEED = np.iinfo(np.int32).max


@spaces.GPU(duration=120)
def generate(
    reference_image,
    prompt=TRIGGER_PROMPT,
    num_inference_steps=8,
    guidance_scale=1.0,
    lora_strength=1.0,
    seed=42,
    randomize_seed=True,
    width=1536,
    height=1024,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate a multi-view character sheet from a single reference image.

    Uses the CharacterSheet QuadView LoRA on FLUX.2 Klein 9B to turn one
    character photo into a face close-up plus front, side, and back full-body
    views arranged on a single sheet.

    Args:
        reference_image: A clear, well-framed image of the character.
        prompt: Instruction caption; defaults to the QuadView trigger.
        num_inference_steps: Sampling steps (8 is the recommended starting point).
        guidance_scale: CFG scale (1.0 for distilled Klein 9B).
        lora_strength: LoRA scale (1.0 = full strength).
        seed: RNG seed for reproducibility.
        randomize_seed: Use a random seed instead of the provided one.
        width: Output width in pixels.
        height: Output height in pixels.
    """
    if reference_image is None:
        raise gr.Error("Please upload a reference character image.")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    seed = int(seed)

    # Apply / adjust LoRA strength
    pipe.set_adapters(["default"], adapter_weights=[float(lora_strength)])

    generator = torch.Generator(device=device).manual_seed(seed)

    img = pipe(
        image=reference_image,
        prompt=prompt,
        width=int(width),
        height=int(height),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
    ).images[0]

    return img, seed


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
CSS = """
#col-container { max-width: 1100px; margin: 0 auto; }
.dark .gradio-container { color: var(--body-text-color); }
"""

with gr.Blocks(theme=gr.themes.Citrus(), css=CSS) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            """# CharacterSheet LoRA — QuadView on FLUX.2 Klein 9B
Turn a single character photo into a multi-view reference sheet (face close-up + front, side, and back full-body views) using the [CharacterSheet LoRA](https://huggingface.co/Alissonerdx/CharacterSheet) on [FLUX.2 Klein 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)."""
        )

        with gr.Row():
            with gr.Column():
                reference_image = gr.Image(
                    label="Reference character image",
                    type="pil",
                    sources=["upload", "clipboard"],
                )
                run_button = gr.Button("Generate character sheet", variant="primary")

                with gr.Accordion("Advanced settings", open=False):
                    prompt = gr.Textbox(
                        label="Instruction prompt",
                        value=TRIGGER_PROMPT,
                        lines=3,
                        info="The trigger caption for the QuadView LoRA. Edit only if you know what you're doing.",
                    )
                    with gr.Row():
                        num_inference_steps = gr.Slider(
                            label="Steps", minimum=1, maximum=20, step=1, value=8,
                            info="8 is the recommended starting point for Klein 9B QuadView.",
                        )
                        guidance_scale = gr.Slider(
                            label="CFG scale", minimum=0.0, maximum=10.0, step=0.1, value=1.0,
                        )
                    lora_strength = gr.Slider(
                        label="LoRA strength", minimum=0.0, maximum=2.0, step=0.05, value=1.0,
                    )
                    with gr.Row():
                        seed = gr.Slider(
                            label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42,
                        )
                        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    with gr.Row():
                        width = gr.Slider(
                            label="Width", minimum=512, maximum=2048, step=16, value=1536,
                        )
                        height = gr.Slider(
                            label="Height", minimum=512, maximum=2048, step=16, value=1024,
                        )

            with gr.Column():
                result = gr.Image(label="Character sheet", show_label=True)

        gr.Examples(
            examples=[
                ["example_ref.jpg", TRIGGER_PROMPT, 8, 1.0, 1.0, 42, True, 1536, 1024],
                ["example_ref_2.jpg", TRIGGER_PROMPT, 8, 1.0, 1.0, 42, True, 1536, 1024],
                ["example_ref_3.jpg", TRIGGER_PROMPT, 8, 1.0, 1.0, 42, True, 1536, 1024],
            ],
            inputs=[
                reference_image, prompt, num_inference_steps, guidance_scale,
                lora_strength, seed, randomize_seed, width, height,
            ],
            outputs=[result, seed],
            fn=generate,
            cache_examples=True,
            cache_mode="lazy",
        )

    gr.on(
        triggers=[run_button.click],
        fn=generate,
        inputs=[
            reference_image, prompt, num_inference_steps, guidance_scale,
            lora_strength, seed, randomize_seed, width, height,
        ],
        outputs=[result, seed],
        api_name="generate",
    )

demo.launch(mcp_server=True)