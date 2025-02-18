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

from typing import Dict

from sqlalchemy import PrimaryKeyConstraint, String, BIGINT, TypeDecorator, TEXT, func, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import Mapped, mapped_column

from repositories.data.database import with_async_session, BasePO, AliasMapper
from repositories.data.message.MessageRepository import (
    MessageRepository,
    MessageSummaryRepository,
)
from repositories.data.message.message_models import (
    MessageBlock,
    Message,
    MessageSummary,
)
from repositories.data.postgres_database import PostgresBasePO
from utils.dictionary import dict_exclude_keys
from utils.json import default_excluded_fields

_alias_mapping: Dict[str, AliasMapper] = {"uid": AliasMapper(alias="message_uid")}


class MessageRepositoryPostgres(MessageRepository):
    """
    会话消息数据存储的Postgres实现
    """

    @staticmethod
    def _convert(message: Message):
        message_po = MessagePO(**dict_exclude_keys(vars(message), ["message_uid", "sender_info"]))
        if message.message_uid:
            message_po.uid = message.message_uid
        else:
            message_po.uid = BasePO.uid_generate()
            message.message_uid = message_po.uid
        return message_po

    @with_async_session
    async def add(self, message: Message, session: AsyncSession) -> Message:
        message_po = MessageRepositoryPostgres._convert(message)
        session.add(message_po)
        message.message_uid = message_po.uid
        return message

    @with_async_session
    async def add_batch(
        self, messages: list[Message], session: AsyncSession
    ) -> list[Message]:
        message_pos = [
            MessageRepositoryPostgres._convert(message) for message in messages
        ]
        session.add_all(message_pos)
        return messages

    @with_async_session
    async def find_latest(
        self,
        conversation_uid: str,
        latest_count: int,
        reset_message_uid: str | None,
        session: AsyncSession,
    ) -> list[Message]:
        stmt = (
            select(MessagePO)
            .where(MessagePO.conversation_uid == conversation_uid)
            .where(MessagePO.is_deleted == False)
        )

        if reset_message_uid:
            reset_message_uid_subquery = (
                select(MessagePO.message_time)
                .where(MessagePO.uid == reset_message_uid)
                .where(MessagePO.is_deleted == False)
            ).scalar_subquery()
            stmt = stmt.where(MessagePO.message_time > reset_message_uid_subquery)

        stmt = (
            stmt.where(MessagePO.is_deleted == False)
            .order_by(MessagePO.message_time.desc())
            .order_by(MessagePO.created_at.desc())
            .limit(latest_count)
        )

        select_result = await session.execute(stmt)
        return [
            Message.model_validate(message.as_dict(alias_mapping=_alias_mapping))
            for message in select_result.scalars()
        ][::-1]

    @with_async_session
    async def find_after_time(
        self,
        conversation_uid: str,
        after_time: int | None,
        max_count: int,
        reset_message_uid: str | None,
        session: AsyncSession,
    ) -> list[Message]:
        stmt = (
            select(MessagePO)
            .where(MessagePO.conversation_uid == conversation_uid)
            .where(MessagePO.is_deleted == False)
        )

        if after_time:
            stmt = stmt.where(MessagePO.message_time > after_time)

        if reset_message_uid:
            reset_message_uid_subquery = (
                select(MessagePO.message_time)
                .where(MessagePO.uid == reset_message_uid)
                .where(MessagePO.is_deleted == False)
            ).scalar_subquery()
            stmt = stmt.where(MessagePO.message_time > reset_message_uid_subquery)

        stmt = (
            stmt.where(MessagePO.is_deleted == False)
            .order_by(MessagePO.message_time.desc())
            .order_by(MessagePO.created_at.desc())
            .limit(max_count)
        )

        select_result = await session.execute(stmt)
        return [
            Message(**message.as_dict(alias_mapping=_alias_mapping))
            for message in select_result.scalars()
        ][::-1]

    @with_async_session
    async def find_after_uid(
        self,
        conversation_uid: str,
        after_uid: str | None,
        include_this: bool,
        max_count: int,
        reset_message_uid: str | None,
        session: AsyncSession,
    ) -> list[Message]:
        stmt = (
            select(MessagePO)
            .where(MessagePO.conversation_uid == conversation_uid)
            .where(MessagePO.is_deleted == False)
        )

        if after_uid:
            after_uid_subquery = (
                select(MessagePO.message_time)
                .where(MessagePO.uid == after_uid)
                .where(MessagePO.is_deleted == False)
            ).scalar_subquery()
            stmt = (
                stmt.where(MessagePO.message_time >= after_uid_subquery)
                if include_this
                else stmt.where(MessagePO.message_time > after_uid_subquery)
            )

        if reset_message_uid:
            reset_message_uid_subquery = (
                select(MessagePO.message_time)
                .where(MessagePO.uid == reset_message_uid)
                .where(MessagePO.is_deleted == False)
            ).scalar_subquery()
            stmt = stmt.where(MessagePO.message_time > reset_message_uid_subquery)

        stmt = (
            stmt.where(MessagePO.is_deleted == False)
            .order_by(MessagePO.message_time.desc())
            .order_by(MessagePO.created_at.desc())
            .limit(max_count)
        )

        select_result = await session.execute(stmt)
        return [
            Message(**message.as_dict(alias_mapping=_alias_mapping))
            for message in select_result.scalars()
        ][::-1]

    @with_async_session
    async def find_before_uid(
        self,
        conversation_uid: str,
        before_uid: str | None,
        include_this: bool,
        max_count: int,
        reset_message_uid: str | None,
        session: AsyncSession,
    ) -> list[Message]:
        stmt = (
            select(MessagePO)
            .where(MessagePO.conversation_uid == conversation_uid)
            .where(MessagePO.is_deleted == False)
        )

        if before_uid:
            before_uid_subquery = (
                select(MessagePO.message_time)
                .where(MessagePO.uid == before_uid)
                .where(MessagePO.is_deleted == False)
            ).scalar_subquery()
            stmt = (
                stmt.where(MessagePO.message_time <= before_uid_subquery)
                if include_this
                else stmt.where(MessagePO.message_time < before_uid_subquery)
            )

        if reset_message_uid:
            reset_message_uid_subquery = (
                select(MessagePO.message_time)
                .where(MessagePO.uid == reset_message_uid)
                .where(MessagePO.is_deleted == False)
            ).scalar_subquery()
            stmt = stmt.where(MessagePO.message_time > reset_message_uid_subquery)

        stmt = stmt.order_by(MessagePO.message_time.desc()).order_by(MessagePO.created_at.desc()).limit(max_count)
        select_result = await session.execute(stmt)
        return [
            Message(**message.as_dict(alias_mapping=_alias_mapping))
            for message in select_result.scalars()
        ][::-1]

    @with_async_session
    async def count_after_uid(
        self,
        conversation_uid: str,
        after_uid: str | None,
        reset_message_uid: str | None,
        session: AsyncSession,
    ) -> int:
        stmt = (
            select(func.count())
            .select_from(MessagePO)
            .where(MessagePO.conversation_uid == conversation_uid)
            .where(MessagePO.is_deleted == False)
        )

        if after_uid:
            after_uid_subquery = (
                select(MessagePO.message_time)
                .where(MessagePO.uid == after_uid)
                .where(MessagePO.is_deleted == False)
            ).scalar_subquery()
            stmt = stmt.where(MessagePO.message_time > after_uid_subquery)

        if reset_message_uid:
            reset_message_uid_subquery = (
                select(MessagePO.message_time)
                .where(MessagePO.uid == reset_message_uid)
                .where(MessagePO.is_deleted == False)
            ).scalar_subquery()
            stmt = stmt.where(MessagePO.message_time > reset_message_uid_subquery)

        count_result = await session.execute(stmt)
        return count_result.scalar()


class MessageSummaryRepositoryPostgres(MessageSummaryRepository):
    @with_async_session
    async def add(
        self, message_summary: MessageSummary, session: AsyncSession
    ) -> MessageSummary:
        message_summary_po = MessageSummaryPO(**vars(message_summary))
        session.add(message_summary_po)
        return message_summary
    
    @with_async_session
    async def disable_all(self, conversation_uid: str, session: AsyncSession) -> bool:
        stmt = (
            update(MessageSummaryPO)
            .where(MessageSummaryPO.conversation_uid == conversation_uid)
            .where(MessageSummaryPO.is_deleted == False)
            .values(is_deleted=True)
        )
        await session.execute(stmt)
        return True

    @with_async_session
    async def get_latest(
        self, conversation_uid: str, session: AsyncSession
    ) -> MessageSummary:
        stmt = (
            select(MessageSummaryPO)
            .where(MessageSummaryPO.conversation_uid == conversation_uid)
            .where(MessageSummaryPO.is_deleted == False)
            .order_by(MessageSummaryPO.summary_order.desc())
            .limit(1)
        )
        select_result = await session.execute(stmt)
        message_summary = select_result.scalars().one_or_none()
        return MessageSummary(**message_summary.as_dict()) if message_summary else None


class MessageBlocks2Jsonb(TypeDecorator):
    impl = JSONB

    def process_bind_param(self, value, dialect):
        """
        Convert Python list[MessageContent] to database jsonb
        """
        if value is not None:
            value = [
                message_block.model_dump(exclude=default_excluded_fields)
                for message_block in value
            ]
        return value

    def process_result_value(self, value, dialect):
        """
        Convert database jsonb to Python list[MessageContent]
        """
        if value is not None:
            value = [MessageBlock.model_validate(json_obj) for json_obj in value]
        return value


class MessagePO(PostgresBasePO):
    """
    消息PO
    """

    __tablename__ = "modu_messages"
    __table_args__ = (PrimaryKeyConstraint("id", name="pk_id"),)

    conversation_uid: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="会话uid"
    )
    message_time: Mapped[int] = mapped_column(
        BIGINT(), nullable=False, comment="消息时间戳"
    )
    sender_uid: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="发送者uid"
    )
    sender_role: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="发送者角色"
    )
    messages: Mapped[list[MessageBlock]] = mapped_column(
        MessageBlocks2Jsonb(), nullable=False, comment="消息"
    )


class MessageSummaryPO(PostgresBasePO):
    """
    消息PO
    """

    __tablename__ = "modu_message_summary"
    __table_args__ = (PrimaryKeyConstraint("id", name="pk_id"),)

    conversation_uid: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="会话uid"
    )
    summary: Mapped[str] = mapped_column(TEXT, nullable=False, comment="会话摘要总结")
    summary_order: Mapped[int] = mapped_column(
        BIGINT(), nullable=False, comment="会话摘要排序"
    )
    last_message_uid: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="会话总结时最后一条消息uid"
    )
