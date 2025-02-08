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

from repositories.cache import cache_decorator_builder
from repositories.cache.cache import CacheDecorator
from repositories.data import account_repository, workspace_repository, database
from repositories.data.workspace.workspace_models import Workspace, WorkspaceType
from utils.auth import hash_password
from utils.common_utils import strtobool
from utils.errors.base_error import ErrorShowType
from utils.errors.system_error import InitializeError

initialized_status_cache: CacheDecorator[bool] = cache_decorator_builder.build(
    serialize=lambda initialized: "True" if initialized else "False",
    deserialize=lambda json_content: bool(strtobool(json_content)),
    default_expire_seconds=30 * 24 * 3600,
    allow_none_values=True,
)


@initialized_status_cache.async_cacheable(
    key_generator=lambda **kwargs: "system:initialized"
)
async def is_initialized() -> bool:
    """
    判断是否已经初始化
    :return: True / False
    """
    return await account_repository.count_all() > 0


@initialized_status_cache.async_cache_evict(
    key_generator=lambda **kwargs: "system:initialized"
)
async def initialize(name: str, email: str, password: str):
    """
    初始化服务
    :param name: 名称
    :param email: 邮箱
    :param password: 密码
    """
    if await is_initialized():
        raise InitializeError(
            message="已初始化，请直接登录",
            status_code=409,
            show_type=ErrorShowType.REDIRECT,
            target="/login",
        )

    # 显式使用session（存在多个写动作，保证事务一致性）
    async with database.async_session() as session:
        # 创建账号
        account = await account_repository.create(
            name, email, hash_password(password), session
        )

        # 创建空间
        await workspace_repository.create(
            Workspace(
                creator_uid=account.uid,
                name="默认空间",
                description="个人默认空间",
                type=WorkspaceType.PRIVATE,
            ),
            session,
        )
