import os
import gc
import time
import torch
import types
import logging
from packaging.version import parse
import sglang
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.model_loader.weight_utils import default_weight_loader
import json

from torch import nn
from .flash_quantization import get_quantize_fn

# Set up logger
logger = logging.getLogger(__name__)

def bond_method_to_cls(func, obj):
    if hasattr(func, '__self__') or not callable(func):
        # If the function is already bound to an instance, return it as is
        return func
    else:
        return types.MethodType(func, obj)

recorded_loader_keys = [
    'weight_loader',
    'load_qkv_weight',
    'load_column_parallel_weight',
    'load_row_parallel_weight',
    'load_merged_column_weight',
    'output_dim',
    'input_dim',
    '_assert_and_load',
]

keys_to_overload = [
    'load_format',
    'quantization',
]

def load_flashrl_config(config):

    config_path = config.strip()

    if config_path in ['bf16', 'fp8', 'fp8_vllm', 'fp8_fast', 'fp8_vllm_fast']:
        logger.info(f"Using profile-free default for: {config_path}")

        from .configs import get_default_config
        from dataclasses import asdict
        config_data = {'configs': [asdict(get_default_config(config_path))]}
    else:
        logger.info(f"Loading flash_rl config from: {config_path}")

        if not os.path.exists(config_path):
            from huggingface_hub import hf_hub_download
            config_path = config_path.split('/')
            assert len(config_path) >= 3, f'Invalid flash_rl config path: {config_path}'
            config_path = hf_hub_download(repo_id='/'.join(config_path[:2]), filename='/'.join(config_path[2:]))

        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

    return config_data

def get_config_data_and_flash_rl_profile():
    config = os.environ.get("FLASHRL_CONFIG", None)
    rank = int(os.environ.get("RANK", None))
    mp_size = int(os.environ.get("MP_SIZE", None))
    dp_rank = rank // mp_size

    if config is not None:
        config_data = load_flashrl_config(config)
        config_count = len(config_data['configs'])
        config_index = dp_rank % config_count
        logger.info(f"Using config {config_index} of {config_count}")
        config_data = config_data['configs'][config_index]
        if config_data.get('fn', 'int8') != 'bf16':
            if config_data.get('fn', 'int8') in ['fp8_vllm', 'fp8', 'fp8_fast', 'fp8_vllm_fast']:
                flash_rl_profile = None
            else:
                model = config_data.get('model', None)
                quant_profile = config_data.get('profile', os.path.join(model, 'profile.pt'))
                logger.debug(f"Loading flash_rl profile from: {quant_profile}")
                flash_rl_profile = torch.load(quant_profile)
        return config_data, flash_rl_profile
    return None, None

def qwen2_get_updated_params(weights, causal_model):
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    params_dict = dict(causal_model.named_parameters())
    updated_params = set()
    is_last_update = False
    for name, loaded_weight in weights:
        if name == "lm_head.weight":
            is_last_update = True

        layer_id = get_layer_id(name)
        if (
            layer_id is not None
            and hasattr(causal_model.model, "start_layer")
            and (
                layer_id < causal_model.model.start_layer
                or layer_id >= causal_model.model.end_layer
            )
        ):
            continue

        if "rotary_emb.inv_freq" in name or "projector" in name:
            continue
        if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
            # Models trained using ColossalAI may include these tensors in
            # the checkpoint. Skip them.
            continue
        if causal_model.config.tie_word_embeddings and "lm_head.weight" in name:
            continue
        if name.startswith("model.vision_tower") and name not in params_dict:
            continue

        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            updated_params.add(name)
            break
        else:
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            if name in params_dict.keys():
                updated_params.add(name)

    return list(updated_params), is_last_update

backup = {}
@staticmethod
def hacked_load_weights_and_postprocess(
    model,
    weights,
    target_device,
    hacked_data_dict = None,
    updated_params = None,
):
    # Hack model.load_weights first.
    if not hasattr(model, 'config_data'):
        config_data, flash_rl_profile = get_config_data_and_flash_rl_profile()
        setattr(model, 'config_data', config_data)
        setattr(model, 'flash_rl_profile', flash_rl_profile)
    if (not hasattr(model, 'beforeflashrl_load_weights')) and (model.config_data.get('fn', 'int8') != 'bf16'):
        # Get quantization function.
        quant_fn = model.config_data.get('fn', 'int8')
        model.flashrl_quant_fn = quant_fn
        logger.debug(f"flash_rl quantization function: {quant_fn}")
        flash_quantize_fn = get_quantize_fn(quant_fn)

        # Store the original load_weights function
        model.beforeflashrl_load_weights = model.load_weights
        def hacked_load_weights(
            self,
            weights,
        ):
            # Skip the case: When reload weights, hacked_load_weights calls load_weights_and_postprocess, it loads weights repeatedly.
            if weights is None:
                return

            start_time = time.time()

            # First time load weights.
            if not hasattr(self, "hacked_original_weights_rebuild_keys"):
                logger.debug("First time load weights, call original_load_weights")
                self.beforeflashrl_load_weights(weights)
                return

            # Get updated params.
            updated_params, is_last_update = qwen2_get_updated_params(weights, self)

            # Record existing_params in hacked_data_dict.
            logger.debug("Run hacked_load_weights, not first time")
            existing_params = dict(self.named_parameters())
            hacked_data_dict = {}
            for name in updated_params:
                hacked_data_dict[name] = existing_params[name].data

            # Clone existing_params and set attributes.
            for name, (shape, stride, dtype, nbytes) in self.hacked_original_weights_rebuild_keys.items():
                if name in updated_params:
                    existing_params[name].data = torch.as_strided(existing_params[name].data.clone(), shape, stride)
            for k, loader_k in self.hacked_recorded_loader.items():
                for n, loader in loader_k.items():
                    if n in updated_params and not hasattr(existing_params[n], k):
                        setattr(existing_params[n], k, bond_method_to_cls(loader, existing_params[n]))
            del existing_params

            end_time = time.time()
            logger.debug(f"flash_rl load_weights preparation took {end_time - start_time:.2f} seconds")
            start_time = end_time

            # Load weights.
            self.beforeflashrl_load_weights(
                flash_quantize_fn(weights, self.flash_rl_profile),
            )
            del weights

            end_time = time.time()
            logger.debug(f"flash_rl original_load_weights took {end_time - start_time:.2f} seconds")
            start_time = end_time

            # Process weights after loading.
            setattr(self, 'hacked_not_need_process_weights_after_loading', True)
            from sglang.srt.model_loader.loader import DefaultModelLoader
            DefaultModelLoader.load_weights_and_postprocess(self, None, None, hacked_data_dict=hacked_data_dict, updated_params=updated_params)

            # Clean up and restore weight_scale.
            del hacked_data_dict
            if is_last_update:
                gc.collect()
                torch.cuda.empty_cache()

                # Restore weight_scale.
                global backup
                for name, p in self.named_parameters():
                    if "weight_scale" in name:
                        p.data.copy_(backup[name])

            end_time = time.time()
            logger.debug(f"flash_rl load_weights process_weights_after_loading took {end_time - start_time:.2f} seconds")
            return

        model.load_weights = types.MethodType(hacked_load_weights, model)
        logger.debug("Successfully patched the load_weights function of sglang")
    else:
        logger.debug("sglang load_weights patching skipped")

    # Record target_device
    if target_device is None:
        target_device = getattr(model, 'hacked_target_device', None)
    else:
        setattr(model, 'hacked_target_device', target_device)

    # Load weights.
    logger.debug("Run hacked_load_weights_and_postprocess")
    model.load_weights(weights)
    original_weights = dict(model.named_parameters())

    # Record original_weights_rebuild_keys.
    # this can be optimized for better memory usage, leave for future work...
    if not hasattr(model, 'hacked_original_weights_rebuild_keys'):
        logger.debug("Record original_weights_rebuild_keys")
        model.hacked_original_weights_rebuild_keys = {}
        for name, p in original_weights.items():
            model.hacked_original_weights_rebuild_keys[name] = (p.shape, p.stride(), p.dtype, p.untyped_storage().nbytes())

    # Record loaders.
    if not hasattr(model, 'hacked_recorded_loader'):
        logger.debug("Record loaders")
        recorded_loader = {k: dict() for k in recorded_loader_keys}
        for name, p in original_weights.items():
            for k in recorded_loader.keys():
                if hasattr(p, k):
                    attr = getattr(p, k)
                    if not callable(attr):
                        recorded_loader[k][name] = attr
                    elif p is attr.__self__:
                        recorded_loader[k][name] = attr.__func__
                    else:
                        recorded_loader[k][name] = attr
        model.hacked_recorded_loader = recorded_loader

    # Original process_weights_after_loading.
    if not getattr(model, 'hacked_not_need_process_weights_after_loading', False):
        logger.debug("Original process weights after loading")
        from sglang.srt.model_loader.loader import device_loading_context
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                # When quant methods need to process weights after loading
                # (for repacking, quantizing, etc), they expect parameters
                # to be on the global target device. This scope is for the
                # case where cpu offloading is used, where we will move the
                # parameters onto device for processing and back off after.
                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)

        # Backup weight_scale after loading weights first time.
        global backup
        processed_weights = dict(model.named_parameters())
        for name, p in processed_weights.items():
            if "weight_scale" in name:
                backup[name] = p.data.clone().cpu()

    # Restore stride and move data back.
    logger.debug(f"hacked_data_dict is None: {hacked_data_dict is None}")
    if hacked_data_dict is not None:
        logger.debug(f"updated_params: {updated_params}")
        processed_weights = dict(model.named_parameters())
        for name in updated_params:
            p = processed_weights[name]
            strided_data = torch.as_strided(p.data, hacked_data_dict[name].shape, hacked_data_dict[name].stride())
            hacked_data_dict[name].copy_(strided_data)
            tmp_data = p.data
            p.data = hacked_data_dict[name]
            del tmp_data

def patch_sglang_load_weights_and_postprocess():
    try:
        from sglang.srt.model_loader.loader import DefaultModelLoader
        if not hasattr(DefaultModelLoader, 'beforeflashrl_load_weights_and_postprocess'):

            original_load_weights_and_postprocess = DefaultModelLoader.load_weights_and_postprocess
            DefaultModelLoader.beforeflashrl_load_weights_and_postprocess = original_load_weights_and_postprocess
            DefaultModelLoader.load_weights_and_postprocess = hacked_load_weights_and_postprocess

            logger.debug("Successfully patched the load_weights_and_postprocess function of sglang")
        else:
            logger.debug("sglang load_weights_and_postprocess already patched")
    except ImportError as e:
        logger.error(f"Error patching sglang load_weights_and_postprocess: {e}")
        return False
    return True

def patch_sglang_Engine():
    try:
        from sglang.srt.entrypoints.engine import Engine
        if not hasattr(Engine, 'beforeflashrl__init__'):
            # Store the original LLM init function
            original_init = Engine.__init__
            Engine.beforeflashrl__init__ = original_init

            def hacked_init_(
                self,
                **kwargs
            ) -> None:
                # Get config path.
                config = os.environ.get("FLASHRL_CONFIG", None)
                assert 'RANK' in os.environ and 'WORLD_SIZE' in os.environ, \
                    'flash_rl only supports external_launcher for now'

                # Get dp rank to calculate config index.
                rank = int(os.environ.get("RANK", None))
                mp_size = kwargs.get('tensor_parallel_size', 1) * kwargs.get('pipeline_parallel_size', 1)
                os.environ['MP_SIZE'] = str(mp_size)
                dp_rank = rank // mp_size

                if config is not None:
                    # Load the config file.
                    # Assuming config is a JSON file, you can use json.load() to read it
                    logger.info(f"flash_rl config detected.")
                    config_data = load_flashrl_config(config)

                    config_count = len(config_data['configs'])
                    config_index = dp_rank % config_count
                    logger.info(f"Using config {config_index} of {config_count}")
                    config_data = config_data['configs'][config_index]

                    for k, v in config_data.items():
                        logger.info(f"rank {rank} flash_rl config: {k}: {v}")

                    # Overload model and other args to engine.
                    for key in keys_to_overload:
                        if key in config_data:
                            logger.debug(f"Overloading {key} with {config_data[key]}")
                            kwargs[key] = config_data.get(key)
                    model = config_data.get('model', kwargs.get('model_path'))
                    kwargs['model_path'] = model
                    
                    # Check model.
                    model_config_path = os.path.join(model, 'config.json')
                    with open(model_config_path, 'r') as f:
                        model_config = json.load(f)
                        model_architectures = model_config.get('architectures', None)
                        if model_architectures:
                            model_architectures = model_architectures[0]
                        logger.debug(f"model_architectures: {model_architectures}")
                        assert model_architectures == 'Qwen2ForCausalLM', "flash_rl only supports Qwen2ForCausalLM for now."
                    
                    if config_data.get('fn', 'int8') != 'bf16':
                        # Check sglang version.
                        if parse(sglang.__version__) != parse('0.4.6.post5'):
                            logger.warning(
                                f'detected sglang version {sglang.__version__}'
                                'for sglang != 0.4.6.post5, `FlashRL` has not been tested'
                            )

                        # Check quantization function.
                        assert config_data.get('fn', 'int8') == 'int8', "flash_rl only supports int8 for sglang"

                        # Set quantization.
                        kwargs['quantization'] = "w8a8_int8"
                else:
                    logger.info(f"flash_rl config not detected.")
                    logger.info(f"Using the original model: {kwargs.get('model')}")

                # Call the parent's __init__ with the custom model
                logger.debug(f"kwargs given to sglang Engine: {kwargs}")
                init_return = original_init(
                    self,
                    **kwargs,
                )

                return init_return

            # Patch the LLM init function
            Engine.__init__ = hacked_init_

            logger.debug("Successfully patched sglang Engine")
        else:
            logger.debug("sglang Engine already patched")
        return True

    except Exception as e:
        logger.error(f"Error patching sglang Engine: {e}")
        return False
