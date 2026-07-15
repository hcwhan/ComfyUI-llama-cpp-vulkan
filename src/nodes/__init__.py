"""节点注册表, 聚合全部节点的 ID -> 类 / 显示名映射."""

from ..i18n.lang import LANG
from .bbox.node_bboxes_to_bbox import bboxes_to_bbox
from .bbox.node_bboxes_to_mask import bboxes_to_mask
from .bbox.node_bboxes_to_segs import bboxes_to_segs
from .bbox.node_json_to_bboxes import json_to_bboxes
from .instruct.media.audio.node_instruct import llama_cpp_audio_instruct
from .instruct.media.image.node_instruct import llama_cpp_image_instruct
from .instruct.media.video.node_instruct import llama_cpp_video_instruct
from .instruct.text.node_instruct import llama_cpp_text_instruct
from .model.node_loaders import llama_cpp_llm_model_loader, llama_cpp_vlm_model_loader
from .model.node_parameters import llama_cpp_parameters
from .model.node_unload import llama_cpp_unload_model
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

# 显示名以语言文件为单一来源; dict() 复制防外部经此引用改写语言字典
NODE_DISPLAY_NAME_MAPPINGS = dict(LANG["display_names"])
