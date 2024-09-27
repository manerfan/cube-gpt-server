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

from __future__ import annotations

from sqlalchemy import PrimaryKeyConstraint, String, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import Mapped, mapped_column

from repositories.data.database import with_async_session
from repositories.data.llm.LLMProviderConfigRepository import LLMProviderConfigRepository
from repositories.data.llm.llm_models import LLMProviderConfig
from repositories.data.postgres_database import PostgresBasePO


class LLMProviderConfigRepositoryPostgres(LLMProviderConfigRepository):
    """
    LLM Provider 配置数据存储的PostgreSQL实现
    """

    @with_async_session
    async def add(
            self, llm_provider_config: LLMProviderConfig, session: AsyncSession) -> LLMProviderConfig:
        provider_config_po = LLMProviderConfigPO.from_llm_provider_config(llm_provider_config)
        session.add(provider_config_po)
        return llm_provider_config

    @with_async_session
    async def update(
            self, llm_provider_config: LLMProviderConfig, session: AsyncSession) -> LLMProviderConfig:
        provider_config = llm_provider_config.copy().encrypt_credential()
        stmt = (update(LLMProviderConfigPO)
                .where(LLMProviderConfigPO.workspace_uid == provider_config.workspace_uid)
                .where(LLMProviderConfigPO.provider_name == provider_config.provider_name)
                .values(provider_credential=provider_config.provider_credential))
        await session.execute(stmt)

        return llm_provider_config

    @with_async_session
    async def delete(self, workspace_uid: str, provider_name: str, session: AsyncSession) -> bool:
        stmt = (delete(LLMProviderConfigPO)
                .where(LLMProviderConfigPO.workspace_uid == workspace_uid)
                .where(LLMProviderConfigPO.provider_name == provider_name))
        await session.execute(stmt)
        return True

    @with_async_session
    async def find_one_by_workspace_and_provider_name(
            self, workspace_uid: str, provider_name: str, session: AsyncSession) -> LLMProviderConfig | None:
        stmt = (select(LLMProviderConfigPO)
                .where(LLMProviderConfigPO.workspace_uid == workspace_uid)
                .where(LLMProviderConfigPO.provider_name == provider_name)
                .where(LLMProviderConfigPO.is_deleted == False)
                .limit(1))
        select_result = await session.execute(stmt)
        provider_config = select_result.scalars().one_or_none()
        if provider_config is None:
            return None

        return provider_config.to_llm_provider_config()

    @with_async_session
    async def list_all(self, workspace_uid: str, session: AsyncSession) -> list[LLMProviderConfig]:
        stmt = (select(LLMProviderConfigPO)
                .where(LLMProviderConfigPO.workspace_uid == workspace_uid)
                .where(LLMProviderConfigPO.is_deleted == False)
                .order_by(LLMProviderConfigPO.created_at.desc()))
        select_result = await session.execute(stmt)
        return [config.to_llm_provider_config() for config in select_result.scalars()]


class LLMProviderConfigPO(PostgresBasePO):
    """
    LLM Provider 配置 PO
    """

    __tablename__ = 'modu_llm_provider_config'
    __table_args__ = (
        PrimaryKeyConstraint('id', name='pk_id'),
    )

    workspace_uid: Mapped[str] = mapped_column(String(32), nullable=False, comment='工作空间uid')
    provider_name: Mapped[str] = mapped_column(String(32), nullable=False, comment='provider key')
    provider_credential: Mapped[str] = mapped_column(String(), nullable=False, comment='provider 凭证')

    @classmethod
    def from_llm_provider_config(cls, llm_provider_config: LLMProviderConfig) -> LLMProviderConfigPO:
        """
        从 LLMProviderConfig 转换
        :return: LLMProviderConfig
        """
        provider_config = llm_provider_config.copy().encrypt_credential()
        return LLMProviderConfigPO(
            workspace_uid=provider_config.workspace_uid,
            provider_name=provider_config.provider_name,
            provider_credential=provider_config.provider_credential,
        )

    def to_llm_provider_config(self) -> LLMProviderConfig:
        """
        转为 LLMProviderConfig
        :return:  LLMProviderConfig
        """
        provider_config = LLMProviderConfig(
            uid=self.uid,
            workspace_uid=self.workspace_uid,
            provider_name=self.provider_name,
            provider_credential=self.provider_credential,
        )
        provider_config.decrypt_credential()

        return provider_config
