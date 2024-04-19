"""
Copyright 2024 Maner·Fan

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from importlib.resources import files
from typing import OrderedDict

import yaml

from utils.reflect.module_scan import load_classes
from .entities.provider import ProviderSchema, ModelProvider


class ModelProviderFactory:
    """
    LLM模型工厂
    """

    _providers: OrderedDict[str, ModelProvider] = {}

    def __init__(self):
        self._load_providers()

    def _load_providers(self):
        """
        加载所有的Provider
        :return:
        """

        if len(self._providers.items()) > 0:
            return

        # 加载provider
        model_provider_cls = load_classes('llm.model.providers', ModelProvider, True, 1)
        model_providers = [provider_cls() for provider_cls in model_provider_cls]

        # 对provider进行排序
        ordinal_content = files('llm.model.providers').joinpath('_ordinal.yml').read_text(encoding='utf-8')
        ordinal_list = yaml.safe_load(ordinal_content)
        ordering = {key: index for index, key in enumerate(ordinal_list)}
        ordering_default = float('inf')
        sorted_providers = sorted(model_providers, key=lambda p: ordering.get(p.provider_schema.key, ordering_default))

        for provider in sorted_providers:
            self._providers[provider.provider_schema.key] = provider

    def get_all_provider_schemas(self) -> list[ProviderSchema]:
        """
        获取所有的Provider Schema
        :return: ProviderSchema
        """
        return [provider.provider_schema for provider in self._providers.values()]