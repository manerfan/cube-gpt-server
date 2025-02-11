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

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from llm.model.entities.models import TextGenerationModel
from utils.dictionary import (
    dict_get,
    dict_exclude_keys,
    dict_map_values,
    dict_filter_none_values,
    dict_merge,
)
from utils.errors.llm_error import LLMValidateError
import openai


class OpenAITextGenerationModel(TextGenerationModel):
    def chat_model(
        self,
        provider_credential: dict,
        model_parameters: dict,
        model_name: str,
        streaming: bool = True,
        request_timeout: int = 10,
        max_retries: int = 0,
    ) -> BaseChatModel:
        # https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format
        model_params = dict_map_values(
            model_parameters, lambda k, v: {"type": v} if k == "response_format" else v
        )

        # see langchain_openai.chat_models.base.BaseChatOpenAI
        # langchain对字段有校验，如果model_params中无参数值传none会报错（应该直接不传），这里通过dict给langchain传值
        model_params = dict_filter_none_values(
            {
                "model": model_name or "gpt-4o",
                "request_timeout": request_timeout,
                "max_retries": max_retries,
                "streaming": streaming,
                "temperature": dict_get(model_params, "temperature"),
                "max_tokens": dict_get(model_params, "max_tokens"),
                "stop": dict_get(model_params, "stop"),
                # 其他模型参数
                "model_kwargs": dict_exclude_keys(
                    model_params, ["request_timeout", "max_retries", "streaming", "temperature", "max_tokens", "stop"]
                ),
            }
        )

        return ChatOpenAI(
            # 模型参数
            **model_params,
            # 认证参数
            **provider_credential,
        )

    async def validate_credentials(
        self, credentials: dict, model: str | None = None
    ) -> None:
        try:
            model_name = model or "gpt-4o"
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
                "OpenAI Credential Validate Success, using model {}, chat result {}",
                model_name,
                chat_result,
            )
        except Exception as e:
            raise LLMValidateError(f"认证异常: {e}")

    def before_invoke(self, messages: list[BaseMessage], model_name: str) -> tuple[list[BaseMessage], dict]:
        if model_name.startswith("o"):
            # o1 之后的模型不再使用system，使用developer替代
            for message in messages:
                if isinstance(message, SystemMessage):
                    # see _convert_message_to_dict
                    message.additional_kwargs = {**message.additional_kwargs, "__openai_role__": "developer"}
        return (messages, {})
