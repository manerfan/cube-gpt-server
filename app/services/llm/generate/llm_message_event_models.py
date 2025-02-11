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

from enum import Enum
from typing import Any, Union

from pydantic import BaseModel
from ulid import ULID
from langchain_core.messages import BaseMessage

from utils.pydantic import default_model_config


class Event(str, Enum):
    MESSAGE_START = "MESSAGE_START"
    """消息开始"""

    MESSAGE_END = "MESSAGE_END"
    """消息结束"""

    AI_MESSAGE_CHUNK = "AI_MESSAGE_CHUNK"
    """AI消息分块"""

    AI_REASONING_CHUNK = "AI_REASONING_CHUNK"
    """AI推理分块"""

    STOP = "STOP"
    """停止"""

    ERROR = "ERROR"
    """异常"""


class MessageEvent(BaseModel):
    """
    消息事件
    """

    event: Event
    """事件类型"""

    section_uid: str
    """内容块ID"""

    # 定义配置
    model_config = default_model_config()


class MessageStartEvent(MessageEvent):
    """
    消息开始事件
    """

    event: Event = Event.MESSAGE_START


class MessageEndEvent(MessageEvent):
    """
    消息结束事件
    """

    event: Event = Event.MESSAGE_END


class StopEvent(MessageEvent):
    """
    停止事件
    """

    event: Event = Event.STOP


class ErrorEvent(MessageEvent):
    """
    异常事件
    """

    event: Event = Event.ERROR

    error: Any | None = None
    """错误异常"""


class AIMessageChunkEvent(MessageEvent):
    """
    AI消息分块事件
    """

    event: Event = Event.AI_MESSAGE_CHUNK

    chunk: Union[str, list[Union[str, dict]]]
    """AI消息分块"""

class AIReasoningChunkEvent(MessageEvent):
    """
    AI推理分块事件
    """

    event: Event = Event.AI_MESSAGE_CHUNK

    reasoning: bool = True
    """AI是否正在推理"""

    chunk: Union[str, list[Union[str, dict]]]
    """AI推理分块"""

class AIReferCardsChunkEvent(MessageEvent):
    """
    引用卡片分块事件
    """

    event: Event = Event.AI_MESSAGE_CHUNK

    chunk: list[dict]
    """引用卡片分块"""
