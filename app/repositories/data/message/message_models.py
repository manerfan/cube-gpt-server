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

from typing import Literal

from pydantic import BaseModel

from utils.pydantic import default_model_config


class MessageBlock(BaseModel):
    """
    消息块
    """

    type: Literal["question", "answer", "system"]
    """
    消息类型
    - question 提问
    - answer 回答
    - system 系统
    """

    content_type: Literal["text", "refer:text", "refer:cards", "think:text", "mention", "error", "system"]
    """
    消息内容的类型
    - text 文本
    - refer:text 引用文本
    - refer:cards 引用卡片
    - think:text 思考文本
    - mention 提及
    - error 错误
    """

    content: str | dict | list[str | dict]
    """消息内容"""

    section_uid: str
    """该部分内容ID"""

    # 定义配置
    model_config = default_model_config()

class MessageBlockChunk(MessageBlock):
    """
    消息块Chunk
    """
    is_finished: bool = False
    """是否结束"""

class SenderInfo(BaseModel):
    """
    发送者信息
    """

    uid: str
    """发送者UID"""

    name: str
    """发送者名称"""

    avatar: str | None = None
    """发送者头像"""

    role: str
    """发送者角色"""

    model_config = default_model_config()


class MessageEventData(BaseModel):
    """
    消息事件
    """

    conversation_uid: str
    """会话ID"""

    message_uid: str
    """消息ID"""

    message_time: int
    """消息时间戳"""

    sender_uid: str
    """发送者UID"""

    sender_info: SenderInfo | None = None
    """发送者信息"""

    sender_role: Literal["user", "assistant", "system"]
    """发送者角色"""

    message: MessageBlockChunk
    """消息内容"""

    is_finished: bool
    """消息是否结束"""

    # 定义配置
    model_config = default_model_config()


class Message(BaseModel):
    """
    消息
    """

    conversation_uid: str
    """会话ID"""

    message_uid: str = ""
    """消息ID"""

    message_time: int
    """消息时间戳"""

    sender_uid: str
    """发送者UID"""

    sender_role: Literal["user", "assistant", "system"]
    """发送者角色"""

    sender_info: SenderInfo | None = None
    """发送者信息"""

    messages: list[MessageBlock]
    """消息内容"""

    # 定义配置
    model_config = default_model_config()


class MessageSummary(BaseModel):
    """
    消息总结
    """

    conversation_uid: str
    """会话ID"""

    summary: str
    """会话摘要总结"""

    summary_order: int
    """会话摘要排序"""

    last_message_uid: str
    """会话总结时最后一条消息uid"""

    # 定义配置
    model_config = default_model_config()
