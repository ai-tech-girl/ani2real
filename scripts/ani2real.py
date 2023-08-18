import gradio as gr
from copy import copy
import importlib
from pathlib import Path
from PIL import Image
import numpy as np

import modules.scripts as scripts
from modules import shared, sd_models, ui, images, generation_parameters_copypaste, sd_samplers_common, script_callbacks
from modules.ui_components import ToolButton
from modules.processing import process_images, program_version, StableDiffusionProcessing, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, Processed
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

preprocessor_sliders_config = {
    "none": [],
    "tile_resample": [
        {
            "label": "Down Sampling Rate",
            "value": 1.0,
            "minimum": 1.0,
            "maximum": 8.0,
            "step": 0.01,
            "visible": True,
            "interactive": True
        }
    ],
    "tile_colorfix": [
        {
            "label": "Variation",
            "value": 8.0,
            "minimum": 3.0,
            "maximum": 32.0,
            "step": 1.0,
            "visible": True,
            "interactive": True
        }
    ],
    "tile_colorfix+sharp": [
        {
            "label": "Variation",
            "value": 8.0,
            "minimum": 3.0,
            "maximum": 32.0,
            "step": 1.0,
            "visible": True,
            "interactive": True
        },
        {
            "label": "Sharpness",
            "value": 1.0,
            "minimum": 0.0,
            "maximum": 2.0,
            "step": 0.01,
            "visible": True,
            "interactive": True
        }
    ]
}

def create_infotext(p, checkpoint_info, all_prompts, all_seeds, all_subseeds, comments=None, iteration=0, position_in_batch=0, use_main_prompt=False, index=None, all_negative_prompts=None):
    if index is None:
        index = position_in_batch + iteration * p.batch_size

    if all_negative_prompts is None:
        all_negative_prompts = p.all_negative_prompts

    clip_skip = getattr(p, 'clip_skip', opts.CLIP_stop_at_last_layers)
    enable_hr = getattr(p, 'enable_hr', False)
    token_merging_ratio = p.get_token_merging_ratio()
    token_merging_ratio_hr = p.get_token_merging_ratio(for_hr=True)

    uses_ensd = opts.eta_noise_seed_delta != 0
    if uses_ensd:
        uses_ensd = sd_samplers_common.is_sampler_using_eta_noise_seed_delta(p)

    generation_params = {
        "Steps": p.steps,
        "Sampler": p.sampler_name,
        "CFG scale": p.cfg_scale,
        "Image CFG scale": getattr(p, 'image_cfg_scale', None),
        "Seed": p.all_seeds[0] if use_main_prompt else all_seeds[index],
        "Face restoration": (opts.face_restoration_model if p.restore_faces else None),
        "Size": f"{p.width}x{p.height}",
        "Model hash": checkpoint_info.hash,
        "Model": checkpoint_info.name_for_extra,
        "Variation seed": (None if p.subseed_strength == 0 else (p.all_subseeds[0] if use_main_prompt else all_subseeds[index])),
        "Variation seed strength": (None if p.subseed_strength == 0 else p.subseed_strength),
        "Seed resize from": (None if p.seed_resize_from_w <= 0 or p.seed_resize_from_h <= 0 else f"{p.seed_resize_from_w}x{p.seed_resize_from_h}"),
        "Denoising strength": getattr(p, 'denoising_strength', None),
        "Conditional mask weight": getattr(p, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) if p.is_using_inpainting_conditioning else None,
        "Clip skip": None if clip_skip <= 1 else clip_skip,
        "ENSD": opts.eta_noise_seed_delta if uses_ensd else None,
        "Token merging ratio": None if token_merging_ratio == 0 else token_merging_ratio,
        "Token merging ratio hr": None if not enable_hr or token_merging_ratio_hr == 0 else token_merging_ratio_hr,
        "Init image hash": getattr(p, 'init_img_hash', None),
        "RNG": opts.randn_source if opts.randn_source != "GPU" else None,
        "NGMS": None if p.s_min_uncond == 0 else p.s_min_uncond,
        **p.extra_generation_params,
        "Version": program_version() if opts.add_version_to_infotext else None,
        "User": p.user if opts.add_user_name_to_info else None,
    }

    generation_params_text = ", ".join([k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in generation_params.items() if v is not None])

    prompt_text = p.prompt if use_main_prompt else all_prompts[index]
    negative_prompt_text = f"\nNegative prompt: {all_negative_prompts[index]}" if all_negative_prompts[index] else ""

    return f"{prompt_text}{negative_prompt_text}\n{generation_params_text}".strip()

def infotext(p, checkpoint_info, index=0, use_main_prompt=False):
    return create_infotext(p, checkpoint_info, p.prompts, p.seeds, p.subseeds, use_main_prompt=use_main_prompt, index=index, all_negative_prompts=p.negative_prompts)

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
                    elem_id="ani2real_enabled",
                )
                save_anime_image = gr.Checkbox(
                    label="Save Anime Image",
                    value=False,
                    visible=True,
                    elem_id="ani2real_save_anime_image",
                )
            with gr.Row():
                ani2real_model_name = gr.Dropdown(sd_models.checkpoint_tiles(), value=lambda: getattr(opts, "ani2real_checkpoint", getattr(opts, "sd_model_checkpoint", sd_models.checkpoint_tiles()[0])), label="Anime model", interactive=True,
                                                  elem_id="ani2real_checkpoint")
                def on_change_model(model_name):
                    opts.set('ani2real_checkpoint', model_name)
                    opts.save(shared.config_filename)
                ani2real_model_name.change(
                    fn=on_change_model,
                    inputs=[ani2real_model_name],
                    outputs=[]
                )
                create_refresh_button(ani2real_model_name, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_ani2real_models")
            with gr.Row():
                prompt = gr.Textbox(
                    show_label=False,
                    lines=3,
                    placeholder="Anime prompt"
                    + "\nIf blank, the main prompt is used.",
                    elem_id="ani2real_prompt",
                )
            with gr.Row():
                negative_prompt = gr.Textbox(
                    show_label=False,
                    lines=2,
                    placeholder="Anime negative prompt"
                    + "\nIf blank, the main prompt is used.",
                    elem_id="ani2real_negative_prompt",
                )
            with gr.Row():
                styles = gr.Dropdown(label="Styles", choices=[k for k, v in shared.prompt_styles.styles.items()], value=[], multiselect=True, elem_id="ani2real_styles")
                clear_prompt_button = ToolButton(value=ui.clear_prompt_symbol, elem_id=f"apply_ani2real_clear_prompt")

                def clear_prompt():
                    return [gr.Textbox.update(value=""), gr.Textbox.update(value="")]

                clear_prompt_button.click(
                    fn=clear_prompt,
                    inputs=[],
                    outputs=[prompt, negative_prompt]
                )
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
                preprocessor = gr.Dropdown(label="ControlNet Preprocessor", choices=preprocessor_sliders_config.keys(), value="none", elem_id="ani2real_preprocessor")
            with gr.Row():
                threshold_a = gr.Slider(visible=False)
                threshold_b = gr.Slider(visible=False)

            def on_change_preprocessor(value):
                sliders_config = preprocessor_sliders_config[value]
                if len(sliders_config) == 0:
                    return [gr.update(visible=False), gr.update(visible=False)]
                elif len(sliders_config) == 1:
                    return [gr.update(**(sliders_config[0])), gr.update(visible=False)]
                else:
                    return [gr.update(**(sliders_config[0])), gr.update(**(sliders_config[1]))]

            preprocessor.change(
                fn=on_change_preprocessor,
                inputs=[preprocessor],
                outputs=[threshold_a, threshold_b]
            )
            with gr.Row():
                weight = gr.Slider(
                    label=f"Control Weight",
                    value=0.4,
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    interactive=True,
                    elem_id="ani2real_control_weight",
                )
                guidance_start = gr.Slider(
                    label="Starting Control Step",
                    value=0,
                    minimum=0.0,
                    maximum=1.0,
                    interactive=True,
                    elem_id="ani2real_guidance_start",
                )
                guidance_end = gr.Slider(
                    label="Ending Control Step",
                    value=0.5,
                    minimum=0.0,
                    maximum=1.0,
                    interactive=True,
                    elem_id="ani2real_ending_control_step",
                )

        self.infotext_fields = [
            (enabled, "Ani2Real Enabled"),
            (ani2real_model_name, "Ani2Real Model"),
            (prompt, "Ani2Real Prompt"),
            (negative_prompt, "Ani2Real Negative Prompt"),
            (preprocessor, "Ani2Real Preprocessor"),
            (threshold_a, "Ani2Real Threshold A"),
            (threshold_b, "Ani2Real Threshold B"),
            (weight, "Ani2Real Weight"),
            (guidance_start, "Ani2Real Guidance Start"),
            (guidance_end, "Ani2Real Guidance End"),
        ]

        elements = [enabled, ani2real_model_name, prompt, negative_prompt, preprocessor, threshold_a, threshold_b, weight, guidance_start, guidance_end, save_anime_image]
        for e in elements:
            setattr(e, "do_not_save_to_config", True)
        return elements

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

    def get_tile_processing(self, p: StableDiffusionProcessing, image: Image, preprocessor, threshold_a, threshold_b, weight, guidance_start, guidance_end):
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
        tile_processing.extra_generation_params = {}

        if isinstance(tile_processing, StableDiffusionProcessingTxt2Img):
            self.enable_cnet_tile(tile_processing, preprocessor, threshold_a, threshold_b, weight, guidance_start, guidance_end, image)
        elif isinstance(tile_processing, StableDiffusionProcessingImg2Img):
            tile_processing.init_images = [image]
            self.enable_cnet_tile(tile_processing, preprocessor, threshold_a, threshold_b, weight, guidance_start, guidance_end)
        else:
            raise RuntimeError("Unsupport processing type")

        return tile_processing

    def enable_cnet_tile(self, p: StableDiffusionProcessing, preprocessor, threshold_a, threshold_b, weight, guidance_start, guidance_end, image=None):
        cnet = find_controlnet()
        tile_model = get_controlnet_tile_model()
        if tile_model:
            tile_units = [cnet.ControlNetUnit(
                module = preprocessor,
                threshold_a = threshold_a,
                threshold_b = threshold_b,
                model = tile_model,
                control_mode = cnet.ControlMode.BALANCED,
                weight = weight,
                guidance_start = guidance_start,
                guidance_end = guidance_end,
                image = np.array(image) if image else None
            )]
            cnet.update_cn_script_in_processing(p, tile_units)

    def before_process(self,
            p: StableDiffusionProcessing,
            enabled: bool,
            ani2real_model_name: str,
            prompt: str,
            negative_prompt: str,
            preprocessor: str,
            threshold_a, threshold_b,
            weight, guidance_start, guidance_end, save_anime_image,
            *args):
        
        if getattr(p, "_disable_ani2real", False):
            return

        if not enabled:
            return

        p._ani2real_origin_processing = copy(p)
        p._ani2real_original_checkpoint_info = p.sd_model.sd_checkpoint_info.name

        # Apply anime prompt
        if len(prompt) > 0:
            p.prompt = prompt
        if len(negative_prompt) > 0:
            p.negative_prompt = negative_prompt

        # Disable HR
        if isinstance(p, StableDiffusionProcessingTxt2Img):
            p.enable_hr = False
        elif isinstance(p, StableDiffusionProcessingImg2Img):
            p.resize_mode = 0
            p.width = p.init_images[0].width
            p.height = p.init_images[0].height
        else:
            raise RuntimeError("Unsupport processing type")

        extra_params = [
            (enabled, "Ani2Real Enabled"),
            (ani2real_model_name, "Ani2Real Model"),
            (prompt, "Ani2Real Prompt"),
            (negative_prompt, "Ani2Real Negative Prompt"),
            (preprocessor, "Ani2Real Preprocessor"),
            (threshold_a, "Ani2Real Threshold A"),
            (threshold_b, "Ani2Real Threshold B"),
            (weight, "Ani2Real Weight"),
            (guidance_start, "Ani2Real Guidance Start"),
            (guidance_end, "Ani2Real Guidance End"),
        ]
        for value, key in extra_params:
            if value is None:
                continue
            if type(value) == 'str' and value == "":
                continue
            p.extra_generation_params[key] = value

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
            preprocessor: str,
            threshold_a, threshold_b,
            weight, guidance_start, guidance_end, save_anime_image):

        if getattr(p, "_disable_ani2real", False):
            return

        if not enabled:
            return

        if shared.state.interrupted or shared.state.skipped:
            # Revert checkpoint
            load_model(p._ani2real_original_checkpoint_info)
            return

        cnet = find_controlnet()

        if not cnet:
            return

        checkpoint_info = sd_models.get_closet_checkpoint_match(model_name)

        text = infotext(p, checkpoint_info)

        p._ani2real_ani_image = copy(pp.image)
        p._ani2real_ani_infotext = text

        if save_anime_image:
            if opts.enable_pnginfo:
                pp.image.info["parameters"] = text
            images.save_image(pp.image, p.outpath_samples, "", p.seeds[p.batch_index], p.prompts[p.batch_index], opts.samples_format, info=text, p=p)

        tile_p = self.get_tile_processing(p, pp.image, preprocessor, threshold_a, threshold_b, weight, guidance_start, guidance_end)
        processed = process_images(tile_p)
        if processed is not None:
            pp.image = processed.images[0]
            p.prompts[p.batch_index] = processed.all_prompts[0]
            p.all_negative_prompts[p.batch_index] = processed.all_negative_prompts[0]

    def postprocess(self, p: StableDiffusionProcessing, processed: Processed, *args):
        if len(processed.images) == 1 and getattr(p, "_ani2real_ani_image", None) and getattr(p, "_ani2real_ani_infotext", None):
            processed.images.extend([
                p._ani2real_ani_image
            ])
            processed.infotexts.extend([
                p._ani2real_ani_infotext
            ])
        if getattr(p, "_ani2real_original_checkpoint_info", None):
            # Revert checkpoint
            load_model(p._ani2real_original_checkpoint_info)

def on_ui_settings():
    section = ("Ani2Real", "Ani2Real")
    shared.opts.add_option(
        "ani2real_checkpoint",
        shared.OptionInfo(
            default=getattr(opts, "sd_model_checkpoint", sd_models.checkpoint_tiles()[0]),
            label="Default model",
            component=gr.Dropdown,
            component_args={
                "choices": sd_models.checkpoint_tiles()
            },
            section=section,
        ),
    )

script_callbacks.on_ui_settings(on_ui_settings)
