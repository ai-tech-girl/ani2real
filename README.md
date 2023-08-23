# Ani2Real extension for Stable diffusion webui

This is an extension for switching checkpoints and performing controlnet's tile in stable diffusion webui.

## Requirements

- stable-diffusion-webui >= 1.4.0
- sd-webui-controlnet >= 1.1

## Install

At first, This extension requires `sd-webui-controlnet`.  
Please install it if you've not yet. see https://github.com/Mikubill/sd-webui-controlnet#installation

1. Open "Extensions" tab.
2. Open "Install from URL" tab in the tab.
3. Enter https://github.com/ai-tech-girl/z_ani2real.git to "URL for extension's git repository".
4. Press "Install" button.

## Specifications

- **Preprocessing**: The checkpoint selected by the `Anime model` is used for the initial processing.
- **Post-processing**: The image generated using the checkpoint from preprocessing and Main is processed by Controlnet tile.
- **Hires** is applied only to post-processing.
- **Adetailer** is applied only to post-processing.
- **Controlnet** that you specified is applied to preprocessing.
- Other plugins might be applied to both preprocessing and post-processing.
- The prompt can be changed for both preprocessing and post-processing.

## Limitations

- Because of the need to switch between two checkpoints, processing will be slower.
