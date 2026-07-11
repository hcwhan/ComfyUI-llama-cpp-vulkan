from .llm import (
    llama_cpp_model_loader,
    llama_cpp_instruct_adv,
    llama_cpp_parameters,
    llama_cpp_unload_model,
)
from .bbox import (
    json_to_bbox,
    bbox_to_segs,
    bbox_to_mask,
    bboxes_to_bbox,
)
from .utils_nodes import (
    parse_json_node,
    remove_code_block,
    PromptEnhancerPreset,
)

NODE_CLASS_MAPPINGS = {
    "llama_cpp_model_loader": llama_cpp_model_loader,
    "llama_cpp_instruct_adv": llama_cpp_instruct_adv,
    "llama_cpp_parameters": llama_cpp_parameters,
    "llama_cpp_unload_model": llama_cpp_unload_model,
    "parse_json_node": parse_json_node,
    "json_to_bbox": json_to_bbox,
    "bbox_to_segs": bbox_to_segs,
    "bbox_to_mask": bbox_to_mask,
    "bboxes_to_bbox": bboxes_to_bbox,
    "remove_code_block": remove_code_block,
    "PromptEnhancerPreset": PromptEnhancerPreset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "llama_cpp_model_loader": "llama.cpp Model Loader",
    "llama_cpp_instruct_adv": "llama.cpp Instruct",
    "llama_cpp_parameters": "llama.cpp Parameters",
    "llama_cpp_unload_model": "llama.cpp Unload Model",
    "parse_json_node": "Parse JSON",
    "json_to_bbox": "JSON to BBoxes",
    "bbox_to_segs": "BBoxes to SEGS",
    "bbox_to_mask": "BBoxes to MASK",
    "bboxes_to_bbox": "BBoxes to BBox",
    "remove_code_block": "Unpack Code Block",
    "PromptEnhancerPreset": "Prompt Enhancer Preset",
}
