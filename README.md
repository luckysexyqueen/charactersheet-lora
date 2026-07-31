---
title: CharacterSheet LoRA Demo
emoji: 🧑‍🎨
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
short_description: Multi-view character sheet from one image (FLUX.2 LoRA)
python_version: "3.12"
startup_duration_timeout: 1h
---

# CharacterSheet LoRA Demo

This Space demonstrates the [CharacterSheet](https://huggingface.co/Alissonerdx/CharacterSheet) QuadView LoRA applied on top of [FLUX.2 Klein 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B).

Upload a clear, well-framed image of a character and the model produces a multi-view character sheet: a face close-up plus front, side, and back full-body views on a single 1536×1024 sheet.