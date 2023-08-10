import gradio as gr
from copy import copy, deepcopy
import importlib
from pathlib import Path
from PIL import Image

import modules.scripts as scripts
from modules import shared, sd_models, ui
from modules.ui_components import ToolButton
from modules.processing import process_images, StableDiffusionProcessing, StableDiffusionProcessingImg2Img, Processed
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

        return [enabled, ani2real_model_name, prompt, negative_prompt, weight, guidance_start, guidance_end]

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

    def script_filter(self, p):
        script_runner = copy(p.scripts)
        script_args = deepcopy(p.script_args)
        self.disable_controlnet_units(script_args)
        return script_runner, script_args

    def disable_controlnet_units(self, script_args):
        for obj in script_args:
            if "controlnet" in obj.__class__.__name__.lower():
                if hasattr(obj, "enabled"):
                    obj.enabled = False
                if hasattr(obj, "input_mode"):
                    obj.input_mode = getattr(obj.input_mode, "SIMPLE", "simple")

            elif isinstance(obj, dict) and "module" in obj:
                obj["enabled"] = False

    def get_tile_i2i(self, p: StableDiffusionProcessing, image: Image):
        seed, subseed = self.get_seed(p, p._ani2real_idx)

        i2i = StableDiffusionProcessingImg2Img(
            init_images=[image],
            resize_mode=0,
            denoising_strength=0.75,
            mask=None,
            mask_blur=None,
            sd_model=p.sd_model,
            outpath_samples=p.outpath_samples,
            outpath_grids=p.outpath_grids,
            prompt=p._ani2real_original_prompt,
            negative_prompt=p._ani2real_original_negative_prompt,
            styles=p.styles,
            seed=seed,
            subseed=subseed,
            subseed_strength=p.subseed_strength,
            seed_resize_from_h=p.seed_resize_from_h,
            seed_resize_from_w=p.seed_resize_from_w,
            sampler_name=p.sampler_name,
            batch_size=1,
            n_iter=1,
            steps=p.steps,
            cfg_scale=p.cfg_scale,
            width=image.width,
            height=image.height,
            restore_faces=p.restore_faces,
            tiling=p.tiling,
            extra_generation_params=p.extra_generation_params,
            override_settings=p.override_settings,
            do_not_save_samples=True,
            do_not_save_grid=True,
        )

        i2i._disable_ani2real = True
        i2i.cached_c = [None, None]
        i2i.cached_uc = [None, None]
        i2i.scripts, i2i.script_args = self.script_filter(p)
        load_model(p._ani2real_original_model_hash)
        return i2i

    def set_cnet_tile(self, p: StableDiffusionProcessing, cnet, weight, guidance_start, guidance_end):
        tile_model = get_controlnet_tile_model()
        if tile_model:
            tile_units = [cnet.ControlNetUnit(
                # module = "tile_resample",
                module = None,
                model = tile_model,
                control_mode = cnet.ControlMode.BALANCED,
                # threshold_a = down_sampling_rate, # Down Sampling Rate
                weight = weight,
                guidance_start = guidance_start,
                guidance_end = guidance_end,
            )]
            cnet.update_cn_script_in_processing(p, tile_units)

    def before_process(self,
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
        
        p._ani2real_original_prompt = p.prompt
        p._ani2real_original_negative_prompt = p.negative_prompt

        if len(prompt) > 0:
            p.prompt = prompt
        if len(negative_prompt) > 0:
            p.negative_prompt = negative_prompt

    def process(self,
            p: StableDiffusionProcessing,
            enabled: bool, *args):

        if getattr(p, "_disable_ani2real", False):
            return
        
        if not enabled:
            return
        
        p._ani2real_original_model_hash = p.sd_model.sd_model_hash

    def process_batch(self,
            p: StableDiffusionProcessing,
            enabled: bool,
            model_name: str, *args, **kwargs):

        if getattr(p, "_disable_ani2real", False):
            return

        if not enabled:
            return

        load_model(model_name)

    def postprocess_image(
            self,
            p: StableDiffusionProcessing,
            pp: scripts.PostprocessImageArgs,
            enabled: bool,
            model_name:str,
            prompt: str,
            negative_prompt: str,
            weight, guidance_start, guidance_end):

        if getattr(p, "_disable_ani2real", False):
            return

        if not enabled:
            return

        cnet = find_controlnet()

        if not cnet:
            return

        p._ani2real_idx = getattr(p, "_ani2real_idx", -1) + 1
        origin = pp.image
        i2i = self.get_tile_i2i(p, origin)
        self.set_cnet_tile(i2i, cnet, weight, guidance_start, guidance_end)
        processed = process_images(i2i)
        if processed is not None:
            pp.image = processed.images[0]
            p._ani2real_origin = origin

    def postprocess(self, p: StableDiffusionProcessing, processed: Processed, *args):
        if len(processed.images) == 1 and getattr(p, "_ani2real_origin", None):
            processed.images.extend([
                p._ani2real_origin
            ])
