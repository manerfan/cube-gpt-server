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

import asyncio

from boltons.strutils import multi_replace
from repositories.data import conversation_repository, message_repository
from repositories.data.account.account_models import Account
from repositories.data.message import ConversationRepository
from repositories.data.message.conversation_models import Conversation
from repositories.data.message.message_models import Message, SenderInfo
from services import account_service, bot_service
from utils.errors.base_error import UnauthorizedError


async def messages(
    current_user: Account,
    conversation_uid: str,
    before_message_uid: str | None,
    max_count: int,
) -> list[Message]:
    """
    查询会话消息
    :param current_user: 当前用户
    :param conversation_uid: 会话 ID
    :param before_message_uid: 查询该消息之前的
    :param max_count: 返回多少条
    """
    conversation = await conversation_repository.get_by_uid(
        current_user.uid, conversation_uid
    )
    if not conversation:
        raise UnauthorizedError("找不到该会话")
    messages = await message_repository.find_before_uid(
        conversation_uid, before_message_uid, False, max_count, None
    )

    # 按照 sender_role 对消息进行分组
    messages_by_role = {}
    for message in messages:
        if message.sender_role not in messages_by_role:
            messages_by_role[message.sender_role] = []
        messages_by_role[message.sender_role].append(message)

    accounts, bots = await asyncio.gather(
        account_service.get_account_infos(
            [message.sender_uid for message in messages_by_role.get("user", [])]
        ),
        bot_service.get_bots_by_uids(
            [message.sender_uid for message in messages_by_role.get("assistant", [])]
        ),
    )

    # 为用户类型的消息填充发送者信息
    for message in messages_by_role.get("user", []):
        if message.sender_uid in accounts:
            account = accounts[message.sender_uid]
            message.sender_info = SenderInfo(
                uid=account.uid,
                name=account.name,
                avatar=getattr(account, "avatar_url", None),
                role="user",
            )

    # 为机器人类型的消息填充发送者信息
    for message in messages_by_role.get("assistant", []):
        if message.sender_uid in bots:
            bot = bots[message.sender_uid]
            message.sender_info = SenderInfo(
                uid=bot.uid,
                name=bot.name,
                avatar=getattr(bot, "avatar_url", None),
                role="assistant",
            )

    return messages


async def conversations(
    current_user: Account, before_conversation_uid: str, max_count: int
) -> list[Conversation]:
    """
    查询会话
    :param before_conversation_uid: 查询该会话之前的
    :param max_count: 返回多少条
    :param current_user: 当前用户
    """
    return await conversation_repository.find_before_uid(
        ConversationRepository.SCOPE_GLOBAL,
        current_user.uid,
        before_conversation_uid,
        False,
        max_count,
    )


async def latest_conversations(current_user: Account) -> Conversation:
    """
    查询最新会话
    :param current_user: 当前用户
    """
    _conversations = await conversation_repository.find_before_uid(
        ConversationRepository.SCOPE_GLOBAL,
        current_user.uid,
        None,
        False,
        1,
    )
    return _conversations[0] if _conversations else None


async def delete_all_conversations(current_user: Account) -> bool:
    """
    删除所有会话
    :param current_user: 当前用户
    """
    return await conversation_repository.delete_all_by_creator(current_user.uid)


async def delete_conversation(current_user: Account, conversation_uid: str) -> bool:
    """
    删除会话
    :param conversation_uid: 会话 ID
    :param current_user: 当前用户
    """
    _conversation = await conversation_repository.get_by_uid(
        current_user.uid, conversation_uid
    )
    if not _conversation:
        raise UnauthorizedError("找不到该会话")

    return await conversation_repository.delete_by_uid(conversation_uid)


async def rename_conversation(
    current_user: Account, conversation_uid: str, name: str
) -> str:
    """
    修改会话名
    :param conversation_uid: 会话 ID
    :param name: 会话名
    :param current_user: 当前用户
    """
    _conversation = await conversation_repository.get_by_uid(
        current_user.uid, conversation_uid
    )
    if not _conversation:
        raise UnauthorizedError("找不到该会话")

    return await conversation_repository.update_name(
        conversation_uid, multi_replace(name, {"\n": " "})[:30]
    )
