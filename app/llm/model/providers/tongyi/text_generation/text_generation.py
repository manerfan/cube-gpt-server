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

from llm.model.providers.tongyi.chat_models import ChatTongyi
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger

from utils.dictionary import (
    dict_exclude_keys,
    dict_filter_none_values,
    dict_get,
    dict_map_values,
    dict_merge,
)
from llm.model.entities.models import TextGenerationModel
from utils.errors.llm_error import LLMValidateError


class DashScopeTextGenerationModel(TextGenerationModel):
    def chat_model(
        self,
        provider_credential: dict,
        model_parameters: dict,
        model_name: str,
        streaming: bool = True,
        request_timeout: int = 10,
        max_retries: int = 0,
    ) -> BaseChatModel:
        # https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-by-calling-api?disableWebsiteRedirect=true#a9b7b197e2q2v
        model_params = dict_map_values(
            model_parameters, lambda k, v: {"type": v} if k == "response_format" else v
        )

        # langchain对字段有校验，如果model_params中无参数值传none会报错（应该直接不传），这里通过dict给langchain传值
        model_params = dict_filter_none_values(
            {
                "model": model_name or "qwen-max",
                "max_retries": max_retries,
                "streaming": streaming,
                "stop": dict_get(model_params, "stop"),
                # 其他模型参数
                "model_kwargs": dict_merge(
                    dict_exclude_keys(model_params, ["max_retries", "streaming", "stop"]),
                    {
                        "request_timeout": request_timeout,
                        # 在流式输出模式下开启增量输出
                        "incremental_output": streaming,
                        # 开启网络搜索
                        **({
                            "search_options" : {
                                "enable_source": True,
                                "enable_citation": True,
                                "citation_format": "[^<number>]",
                                "search_strategy": "standard"
                            }
                        } if "enable_search" in model_params and model_parameters["enable_search"] == True else {})
                    },
                ),
            }
        )

        return ChatTongyi(
            # 模型参数
            **model_params,
            # 认证参数
            **provider_credential,
        )

    async def validate_credentials(
        self, credentials: dict, model: str | None = None
    ) -> None:
        try:
            model_name = model or "qwen-turbo"
            chat_model = self.chat_model(
                provider_credential=credentials,
                model_parameters={"max_tokens": 512},
                model_name=model_name,
                streaming=False,
                request_timeout=10,
                max_retries=3,
            )
            chat_result = await chat_model.ainvoke(
                [
                    SystemMessage(
                        content="You are an AI assistant designed to test the API. Your task is to simply return the user's original input verbatim, without any modifications or additional text."
                    ),
                    HumanMessage(content="林中通幽境，深山藏小舍"),
                ]
            )
            logger.info(
                "Tongyi Credential Validate Success, using model {}, chat result {}",
                model_name,
                chat_result,
            )
        except Exception as e:
            raise LLMValidateError(f"认证异常: {e}")
