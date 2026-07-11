"""GGUF 模型目录注册与路径查找, 注册 llm/LLM 两个目录键, 支持 extra_model_paths.yaml 挂载."""

import os

import folder_paths

llm_extensions = {'.gguf'}
_LLM_FOLDER_KEYS = ("llm", "LLM")
for _key in _LLM_FOLDER_KEYS:
    folder_paths.add_model_folder_path(_key, os.path.join(folder_paths.models_dir, _key))
    folder_paths.folder_names_and_paths[_key][1].update(llm_extensions)


def get_llm_filename_list():
    seen = set()
    result = []
    for key in _LLM_FOLDER_KEYS:
        if key in folder_paths.folder_names_and_paths:
            for f in folder_paths.get_filename_list(key):
                if f not in seen:
                    seen.add(f)
                    result.append(f)
    return result


def get_llm_full_path(filename):
    for key in _LLM_FOLDER_KEYS:
        if key in folder_paths.folder_names_and_paths:
            path = folder_paths.get_full_path(key, filename)
            if path is not None:
                return path
    return None
