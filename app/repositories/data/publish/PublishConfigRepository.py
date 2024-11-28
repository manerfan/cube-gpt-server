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
from repositories.data.publish.publish_config import PublishConfig


class PublishConfigRepository(Repository):
    """
    配置发布数据存储的定义
    """

    def __init__(self, database: Database):
        super().__init__(database)

    @abstractmethod
    async def add_or_update(self, publish_config: PublishConfig, session: AsyncSession) -> PublishConfig:
        """
        新增/更新发布配置
        :param publish_config: 配置内容
        :param session: Session
        """
        raise NotImplementedError()


