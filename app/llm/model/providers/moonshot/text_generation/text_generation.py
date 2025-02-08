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

from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger

from llm.model.entities.models import TextGenerationModel
from utils.errors.llm_error import LLMValidateError


class MoonShotTextGenerationModel(TextGenerationModel):
    def chat_model(
        self,
        provider_credential: dict,
        model_parameters: dict,
        model_name: str,
        streaming: bool = True,
        request_timeout: int = 10,
        max_retries: int = 0,
    ) -> BaseChatModel:
        pass

    async def validate_credentials(
        self, credentials: dict, model: str | None = None
    ) -> None:
        try:
            model_name = model or "moonshot-v1-32k"
            chat_model = MoonshotChat(
                model_name=model_name,
                request_timeout=5,
                max_retries=0,
                max_tokens=512,
                streaming=False,
                **credentials,
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
                "MoonShot Credential Validate Success, using model {}, chat result {}",
                model_name,
                chat_result,
            )
        except Exception as e:
            raise LLMValidateError(f"认证异常: {e}")
