import gradio as gr
from copy import copy
import importlib
from pathlib import Path
from PIL import Image
import numpy as np

import modules.scripts as scripts
from modules import shared, sd_models, ui, images
from modules.ui_components import ToolButton
from modules.processing import process_images, create_infotext, StableDiffusionProcessing, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, Processed
from modules.ui import create_refresh_button
from modules.paths import models_path
from modules.shared import opts

def get_controlnet_model_dirs():
    dirs = [Path(models_path, "ControlNet")]
    for ext_dir in [shared.opts.data.get("control_net_models_path", ""), getattr(shared.cmd_opts, "controlnet_dir", "")]:
        if ext_dir:
            dirs.append(Path(ext_dir))

    return dirs

def get_controlnet_tile_model():
    model_extensions = (".pt", ".pth", ".ckpt", ".safetensors")
    dirs = get_controlnet_model_dirs()
    name_filter = shared.opts.data.get("control_net_models_name_filter", "")
    name_filter = name_filter.strip(" ").lower()

    for base in dirs:
        if not base.exists():
            continue

        for p in base.rglob("*"):
            if (
                p.is_file()
                and p.suffix in model_extensions
                and "tile" in p.name
            ):
                if name_filter and name_filter not in p.name.lower():
                    continue
                model_hash = sd_models.model_hash(p)
                return f"{p.stem} [{model_hash}]"
    return None

def find_controlnet():
    try:
        cnet = importlib.import_module("extensions.sd-webui-controlnet.scripts.external_code")
    except Exception:
        try:
            cnet = importlib.import_module("extensions-builtin.sd-webui-controlnet.scripts.external_code")
        except Exception:
            cnet = None

    return cnet

def load_model(model_name: str):
    info = sd_models.get_closet_checkpoint_match(model_name)
    if info is None:
        raise RuntimeError(f"Unknown checkpoint: {model_name}")
    sd_models.reload_model_weights(info=info)

def infotext(p, index=0, use_main_prompt=False):
    return create_infotext(p, p.prompts, p.seeds, p.subseeds, use_main_prompt=use_main_prompt, index=index, all_negative_prompts=p.negative_prompts)

class Ani2Real(scripts.Script):
    def __init__(self):
        super().__init__()

    def title(self):
        return "Ani2Real"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Ani2Real", open=False):
            with gr.Row():
                enabled = gr.Checkbox(
                    label="Enabled",
                    value=False,
                    visible=True,
                )
            with gr.Row():
                ani2real_model_name = gr.Dropdown(sd_models.checkpoint_tiles(), value=getattr(opts, "sd_model_checkpoint", sd_models.checkpoint_tiles()[0]), label="Anime checkpoint", interactive=True)
                create_refresh_button(ani2real_model_name, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_ani2real_models")
            with gr.Row():
                prompt = gr.Textbox(
                    show_label=False,
                    lines=3,
                    placeholder="Anime prompt"
                    + "\nIf blank, the main prompt is used."
                )
            with gr.Row():
                negative_prompt = gr.Textbox(
                    show_label=False,
                    lines=2,
                    placeholder="Anime negative prompt"
                    + "\nIf blank, the main prompt is used."
                )
            with gr.Row():
                styles = gr.Dropdown(label="Styles", choices=[k for k, v in shared.prompt_styles.styles.items()], value=[], multiselect=True)
                apply_button = ToolButton(value=ui.apply_style_symbol, elem_id="apply_ani2real_styles")

                def apply_styles(prompt, negative_prompt, styles):
                    prompt = shared.prompt_styles.apply_styles_to_prompt(prompt, styles)
                    negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(negative_prompt, styles)
                    return [gr.Textbox.update(value=prompt), gr.Textbox.update(value=negative_prompt), gr.Dropdown.update(value=[])]

                apply_button.click(
                    fn=apply_styles,
                    inputs=[prompt, negative_prompt, styles],
                    outputs=[prompt, negative_prompt, styles],
                )
                create_refresh_button(styles, shared.prompt_styles.reload, lambda: {"choices": [k for k, v in shared.prompt_styles.styles.items()]}, "refresh_ani2real_styles")
            with gr.Row():
                weight = gr.Slider(
                    label=f"Control Weight",
                    value=0.4,
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                )
                guidance_start = gr.Slider(
                    label="Starting Control Step",
                    value=0,
                    minimum=0.0,
                    maximum=1.0,
                    interactive=True,
                )
                guidance_end = gr.Slider(
                    label="Ending Control Step",
                    value=0.5,
                    minimum=0.0,
                    maximum=1.0,
                    interactive=True,
                )
            with gr.Row():
                save_anime_image = gr.Checkbox(
                    label="Save Anime Image",
                    value=False,
                    visible=True,
                )
        return [enabled, ani2real_model_name, prompt, negative_prompt, weight, guidance_start, guidance_end, save_anime_image]

    def get_seed(self, p, idx):
        if not p.all_seeds:
            seed = p.seed
        elif idx < len(p.all_seeds):
            seed = p.all_seeds[idx]
        else:
            j = idx % len(p.all_seeds)
            seed = p.all_seeds[j]

        if not p.all_subseeds:
            subseed = p.subseed
        elif idx < len(p.all_subseeds):
            subseed = p.all_subseeds[idx]
        else:
            j = idx % len(p.all_subseeds)
            subseed = p.all_subseeds[j]

        return seed, subseed

    def get_tile_processing(self, p: StableDiffusionProcessing, image: Image, weight, guidance_start, guidance_end):
        p._ani2real_idx = getattr(p, "_ani2real_idx", -1) + 1
        seed, subseed = self.get_seed(p, p._ani2real_idx)
        tile_processing = copy(p._ani2real_origin_processing)
        tile_processing._disable_ani2real = True
        tile_processing.seed = seed
        tile_processing.subseed = subseed
        tile_processing.batch_size = 1
        tile_processing.n_iter = 1
        tile_processing.do_not_save_samples = True
        tile_processing.do_not_save_grid = True
        tile_processing.cached_c = [None, None]
        tile_processing.cached_uc = [None, None]

        if isinstance(tile_processing, StableDiffusionProcessingTxt2Img):
            self.enable_cnet_tile(tile_processing, weight, guidance_start, guidance_end, image)
        elif isinstance(tile_processing, StableDiffusionProcessingImg2Img):
            tile_processing.init_images = [image]
            self.enable_cnet_tile(tile_processing, weight, guidance_start, guidance_end)
        else:
            raise RuntimeError("Unsupport processing type")

        return tile_processing

    def enable_cnet_tile(self, p: StableDiffusionProcessing, weight, guidance_start, guidance_end, image=None):
        cnet = find_controlnet()
        tile_model = get_controlnet_tile_model()
        if tile_model:
            tile_units = [cnet.ControlNetUnit(
                module = None,
                model = tile_model,
                control_mode = cnet.ControlMode.BALANCED,
                weight = weight,
                guidance_start = guidance_start,
                guidance_end = guidance_end,
                image = np.array(image) if image else None
            )]
            cnet.update_cn_script_in_processing(p, tile_units)

    def process(self,
            p: StableDiffusionProcessing,
            enabled: bool,
            model_name:str,
            prompt: str,
            negative_prompt: str,
            *args):
        
        if getattr(p, "_disable_ani2real", False):
            return
        
        if not enabled:
            return
        
        p._ani2real_origin_processing = copy(p)
        p._ani2real_original_model_hash = p.sd_model.sd_model_hash

        # Apply anime prompt
        if len(prompt) > 0:
            p.prompt = prompt
        if len(negative_prompt) > 0:
            p.negative_prompt = negative_prompt

    def process_batch(self,
            p: StableDiffusionProcessing,
            enabled: bool,
            model_name: str, *args, **kwargs):

        if getattr(p, "_disable_ani2real", False):
            return

        if not enabled:
            return

        # Disable HR
        if isinstance(p, StableDiffusionProcessingTxt2Img):
            p.enable_hr = False
        elif isinstance(p, StableDiffusionProcessingImg2Img):
            p.resize_mode = 0
            p.width = p.init_images[p.iteration].width
            p.height = p.init_images[p.iteration].height
        else:
            raise RuntimeError("Unsupport processing type")

        load_model(model_name)

    def postprocess_image(
            self,
            p: StableDiffusionProcessing,
            pp: scripts.PostprocessImageArgs,
            enabled: bool,
            model_name:str,
            prompt: str,
            negative_prompt: str,
            weight, guidance_start, guidance_end, save_anime_image):

        if getattr(p, "_disable_ani2real", False):
            return

        if not enabled:
            return

        cnet = find_controlnet()

        if not cnet:
            return

        tile_p = self.get_tile_processing(p, pp.image, weight, guidance_start, guidance_end)
        load_model(p._ani2real_original_model_hash)
        processed = process_images(tile_p)
        if processed is not None:
            p._ani2real_anime_image = pp.image
            info = infotext(p, p.batch_index)
            p._ani2real_anime_infotext = info

            if save_anime_image:
                images.save_image(pp.image, p.outpath_samples, "",
                    tile_p.seed, tile_p.prompt, opts.samples_format, info=info, p=p)

            pp.image = processed.images[0]

    def postprocess(self, p: StableDiffusionProcessing, processed: Processed, *args):
        if len(processed.images) == 1 and getattr(p, "_ani2real_anime_image", None) and getattr(p, "_ani2real_anime_infotext", None):
            processed.images.extend([
                p._ani2real_anime_image
            ])
            processed.infotexts.extend([
                p._ani2real_anime_infotext
            ])
