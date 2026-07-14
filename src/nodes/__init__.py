"""节点注册表, 聚合全部节点的 ID -> 类 / 显示名映射."""

from .model.node_loaders import llama_cpp_llm_model_loader, llama_cpp_vlm_model_loader
from .model.node_parameters import llama_cpp_parameters
from .model.node_unload import llama_cpp_unload_model
from .type.text.node_instruct import llama_cpp_text_instruct
from .type.media.image.node_instruct import llama_cpp_image_instruct
from .type.media.video.node_instruct import llama_cpp_video_instruct
from .type.media.audio.node_instruct import llama_cpp_audio_instruct
from .type.media.bbox.node_bbox import (
    json_to_bboxes,
    bboxes_to_segs,
    bboxes_to_mask,
    bboxes_to_bbox,
)
from .util.node_parse_json import parse_json_node
from .util.node_remove_code_block import remove_code_block
from .util.node_split_output import split_instruct_output
from .util.node_system_prompt import system_prompt_preset

NODE_CLASS_MAPPINGS = {
    "llama_cpp_llm_model_loader": llama_cpp_llm_model_loader,
    "llama_cpp_vlm_model_loader": llama_cpp_vlm_model_loader,
    "llama_cpp_parameters": llama_cpp_parameters,
    "llama_cpp_unload_model": llama_cpp_unload_model,
    "llama_cpp_text_instruct": llama_cpp_text_instruct,
    "llama_cpp_image_instruct": llama_cpp_image_instruct,
    "llama_cpp_video_instruct": llama_cpp_video_instruct,
    "llama_cpp_audio_instruct": llama_cpp_audio_instruct,
    "json_to_bboxes": json_to_bboxes,
    "bboxes_to_segs": bboxes_to_segs,
    "bboxes_to_mask": bboxes_to_mask,
    "bboxes_to_bbox": bboxes_to_bbox,
    "parse_json_node": parse_json_node,
    "remove_code_block": remove_code_block,
    "split_instruct_output": split_instruct_output,
    "system_prompt_preset": system_prompt_preset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "llama_cpp_llm_model_loader": "llama.cpp llm Model Loader",
    "llama_cpp_vlm_model_loader": "llama.cpp vlm Model Loader",
    "llama_cpp_parameters": "llama.cpp Parameters",
    "llama_cpp_unload_model": "llama.cpp Unload Model",
    "llama_cpp_text_instruct": "llama.cpp text Instruct",
    "llama_cpp_image_instruct": "llama.cpp image Instruct",
    "llama_cpp_video_instruct": "llama.cpp video Instruct",
    "llama_cpp_audio_instruct": "llama.cpp audio Instruct",
    "json_to_bboxes": "JSON to BBoxes",
    "bboxes_to_segs": "BBoxes to SEGS",
    "bboxes_to_mask": "BBoxes to MASK",
    "bboxes_to_bbox": "BBoxes to BBox",
    "parse_json_node": "Parse JSON",
    "remove_code_block": "Unpack Code Block",
    "split_instruct_output": "Split Instruct Output",
    "system_prompt_preset": "System Prompt Preset",
}
