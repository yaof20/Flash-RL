import logging
import os

# Get logging configuration from environment
log_file = os.getenv("FLASHRL_LOGGING_FILE")
log_level = os.getenv("FLASHRL_LOGGING_LEVEL", "INFO")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, log_level.upper()))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

if log_file:
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
logger = logging.getLogger(__name__)

def check_vllm_installed():
    """Check if vllm is installed"""
    try:
        import vllm
        return True
    except ImportError:
        return False

def check_sglang_installed():
    """Check if sglang is installed"""
    try:
        import sglang
        return True
    except ImportError:
        return False

def check_dist_initialized():
    """Check if distributed environment is initialized"""
    try:
        from torch.distributed import is_initialized
        return is_initialized()
    except ImportError:
        return False

# Check if patching is needed based on environment variables
if 'FLASHRL_CONFIG' in os.environ and 'SGLANG_PATCH' not in os.environ and check_vllm_installed():
    
    from .vllm_patch import patch_vllm_llm, patch_vllm_process_weights_after_loading, patch_vllm_fp8_create_weight 

    # Patch the process_weights_after_loading function
    process_weights_status = patch_vllm_process_weights_after_loading()
    logger.debug(f"Patching vllm process_weights_after_loading... status: {process_weights_status}")
    
    # Patch the LLM class
    patch_status = patch_vllm_llm()
    logger.debug(f"Patching the vllm LLM to enable flash_rl quantization... status: {patch_status}")
    
    patch_vllm_fp8_create_weight_status = patch_vllm_fp8_create_weight()
    logger.debug(f"Patching the vllm fp8 linear... status: {patch_vllm_fp8_create_weight_status}")

    if 'FLASHRL_TEST_RELOAD' in os.environ:
        from .vllm_patch import patch_vllm_llm_test_reload
        reload_status = patch_vllm_llm_test_reload()
        logger.debug(f"Patching vllm LLM init to test reload... status: {reload_status}")

    if os.environ.get('FLASHRL_LMHEAD_FP32', '0') == '1':
        from .vllm_patch import patch_vllm_lmhead_to_fp32
        patch_status = patch_vllm_lmhead_to_fp32()
        logger.debug(f"Patching vllm lmhead to fp32... status: {patch_status}")
elif 'FLASHRL_CONFIG' in os.environ and 'SGLANG_PATCH' in os.environ and check_sglang_installed():
    from .sglang_patch import patch_sglang_Engine, patch_sglang_load_weights_and_postprocess
    patch_status = patch_sglang_Engine()
    logger.debug(f"Patching the sglang Engine status: {patch_status}")

    patch_status = patch_sglang_load_weights_and_postprocess()
    logger.debug(f"Patching the sglang load_weights_and_postprocess status: {patch_status}")
else:
    logger.debug("Skipping the patching of vllm")