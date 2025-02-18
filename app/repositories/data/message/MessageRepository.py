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

from abc import abstractmethod

from sqlalchemy.ext.asyncio import AsyncSession

from repositories.data.database import Repository, Database
from repositories.data.message.message_models import Message, MessageSummary


class MessageRepository(Repository):
    """
    会话消息数据存储的定义
    """

    def __init__(self, database: Database):
        super().__init__(database)

    @abstractmethod
    async def add(self, message: Message, session: AsyncSession) -> Message:
        """
        新增会话消息
        :param message: 会话消息
        :param session: Session
        """
        raise NotImplementedError()

    @abstractmethod
    async def add_batch(
        self, messages: list[Message], session: AsyncSession
    ) -> list[Message]:
        """
        批量新增会话消息
        :param messages: 会话消息
        :param session: Session
        """
        raise NotImplementedError()

    @abstractmethod
    async def find_latest(
        self,
        conversation_uid: str,
        latest_count: int,
        reset_message_uid: str,
        session: AsyncSession,
    ) -> list[Message]:
        """
        查找最新的 n 条消息
        :param conversation_uid: 会话ID
        :param latest_count: 最新消息条数
        :param reset_message_uid: 重置消息 UID
        :param session: Session
        """
        raise NotImplementedError()

    @abstractmethod
    async def find_after_time(
        self,
        conversation_uid: str,
        after_time: int,
        max_count: int,
        reset_message_uid: str,
        session: AsyncSession,
    ) -> list[Message]:
        """
        查找某时间之后的所有消息
        :param conversation_uid: 会话ID
        :param after_time: 某时间后的消息
        :param max_count: 返回的最大条数
        :param reset_message_uid: 重置消息 UID
        :param session: Session
        """
        raise NotImplementedError()

    @abstractmethod
    async def find_after_uid(
        self,
        conversation_uid: str,
        after_uid: str,
        include_this: bool,
        max_count: int,
        reset_message_uid: str,
        session: AsyncSession,
    ) -> list[Message]:
        """
        查找某条消息之后的所有消息
        :param conversation_uid: 会话ID
        :param after_uid: 某消息uid
        :param include_this: 是否包含该条
        :param max_count: 返回的最大条数
        :param reset_message_uid: 重置消息 UID
        :param session: Session
        """
        raise NotImplementedError()

    @abstractmethod
    async def find_before_uid(
        self,
        conversation_uid: str,
        before_uid: str,
        include_this: bool,
        max_count: int,
        reset_message_uid: str,
        session: AsyncSession,
    ) -> list[Message]:
        """
        查找某条消息之前的所有消息
        :param conversation_uid: 会话ID
        :param before_uid: 某消息uid
        :param include_this: 是否包含该条
        :param max_count: 返回的最大条数
        :param reset_message_uid: 重置消息 UID
        :param session: Session
        """
        raise NotImplementedError()

    @abstractmethod
    async def count_after_uid(
        self,
        conversation_uid: str,
        after_uid: str | None,
        reset_message_uid: str,
        session: AsyncSession,
    ) -> int:
        """
        查找某条消息之后的所有消息的数量
        :param conversation_uid: 会话ID
        :param after_uid: 某消息uid
        :param reset_message_uid: 重置消息 UID
        :param session: Session
        """
        raise NotImplementedError()


class MessageSummaryRepository(Repository):
    """
    会话消息总结数据存储的定义
    """

    def __init__(self, database: Database):
        super().__init__(database)

    @abstractmethod
    async def add(
        self, message_summary: MessageSummary, session: AsyncSession
    ) -> MessageSummary:
        """
        新增会话消息总结
        :param message_summary: 会话消息总结
        :param session: Session
        """
        raise NotImplementedError()
    
    @abstractmethod
    async def disable_all(self, conversation_uid: str, session: AsyncSession) -> bool:
        """
        禁用所有会话消息总结
        """
        raise NotImplementedError()

    @abstractmethod
    async def get_latest(self, conversation_uid: str, session: AsyncSession) -> MessageSummary:
        """
        获取最新的会话消息总结
        :param conversation_uid: 会话 ID
        """
        raise NotImplementedError()
