# Copyright (C) 2025 Xiaomi Corporation.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import torch
from einops import rearrange
from model_dasheng import MusicInstrumentRecognize_Dasheng
from netease.utils import load_config, save_config
from thop import profile
from torchinfo import summary


MODEL_MAP = {
    "dasheng_base": MusicInstrumentRecognize_Dasheng
}


def create_model(model_name: str, **kwargs):
    """
    根据模型名称创建对应模型实例
    :param model_name: 配置文件中定义的模型名称
    :param kwargs: 模型初始化参数（如hidden_dims等）
    """
    model_class = MODEL_MAP.get(model_name.lower())

    if not model_class:
        available_models = ", ".join(MODEL_MAP.keys())
        raise ValueError(f"Invalid model name: {model_name}. Available models: {available_models}")

    return model_class(**kwargs)


