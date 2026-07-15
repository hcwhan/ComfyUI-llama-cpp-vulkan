"""GGUF 模型目录注册与路径查找, 注册 llm/LLM 两个目录键, 支持 extra_model_paths.yaml 挂载."""

import os

import folder_paths

llm_extensions = {".gguf"}
_LLM_FOLDER_KEYS = ("llm", "LLM")
for _key in _LLM_FOLDER_KEYS:
    folder_paths.add_model_folder_path(_key, os.path.join(folder_paths.models_dir, _key))
    _paths, _exts = folder_paths.folder_names_and_paths[_key]
    # 恒重建扩展名集合 (复制合并后整体替换本键元组), 不原位 update: 键可能
    # 已被其他插件注册, 其集合可能与 checkpoints/loras 等约 20 个内置键共享
    # 同一 set 实例 (supported_pt_extensions), 原位写入会把 .gguf 泄漏进全部
    # 内置下拉框. set(_exts) 同时兼容第三方以 list/tuple 形态注册的键;
    # _paths 沿用原列表引用, extra_model_paths.yaml 的后续注册不受影响
    folder_paths.folder_names_and_paths[_key] = (_paths, set(_exts) | llm_extensions)


def get_llm_filename_list():
    merged = set()
    for key in _LLM_FOLDER_KEYS:
        if key in folder_paths.folder_names_and_paths:
            for f in folder_paths.get_filename_list(key):
                # llm/LLM 键的扩展名集合是全局共享的, 其他插件可能追加非 gguf
                # 格式, 这里只保留本插件能加载的 .gguf
                if f.lower().endswith(".gguf"):
                    merged.add(f)
    # 跨键合并后统一排序, 与 folder_paths.get_filename_list 的单键排序规则
    # 一致; 两键为独立目录时 (Linux 下 llm/LLM 大小写不同名), 按键拼接会使
    # 下拉框顺序割裂
    return sorted(merged)


def get_llm_full_path(filename):
    for key in _LLM_FOLDER_KEYS:
        if key in folder_paths.folder_names_and_paths:
            path = folder_paths.get_full_path(key, filename)
            if path is not None:
                return path
    return None
