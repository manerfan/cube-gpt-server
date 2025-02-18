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

from datetime import datetime
from typing import AsyncIterator

from boltons.strutils import multi_replace
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.schema import StreamEvent
from langgraph.constants import START
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.graph.graph import CompiledGraph
from loguru import logger
from pydantic import BaseModel
from starlette.responses import ContentStream
from ulid import ULID

from repositories.data.bot.bot_models import BotMode
from services import bot_service
from llm.model import model_provider_factory
from llm.model.entities.model import ModelType
from llm.model.entities.models import TextGenerationModel
from repositories.data import conversation_repository, message_repository, message_summary_repository
from repositories.data.account.account_models import Account
from repositories.data.database import BasePO
from repositories.data.message.conversation_models import Conversation
from repositories.data.message.message_models import (
    MessageBlock,
    MessageBlockChunk,
    MessageEventData,
    Message,
    SenderInfo,
)
from services.llm import llm_model_service, llm_provider_service
from services.llm.generate.llm_message_checkpoint import LLMMessageCheckPointSaver
from services.llm.generate.llm_message_event_models import (
    AIReferCardsChunkEvent,
    AIReasoningChunkEvent,
    MessageStartEvent,
    MessageEndEvent,
    ErrorEvent,
    MessageEvent,
    AIMessageChunkEvent,
)
from services.llm.generate.llm_message_memory import ConversationSummaryBufferMemory
from utils.dictionary import dict_get, dict_merge
from utils.errors.base_error import UnauthorizedError
from utils.errors.llm_error import LLMExistsError
from utils.iterator import single_element_async_iterator, merge_async_iterators
from utils.pydantic import default_model_config
from cachetools import TTLCache

from langgraph.prebuilt import create_react_agent

# 记录正在运行的generator，用于控制是否要中断运行
generate_runner = TTLCache(maxsize=64, ttl=10 * 60)


class Item(BaseModel):
    type: str
    """类型"""

    content: str | dict
    """内容"""

    # 设置允许额外字段
    # 定义配置
    model_config = default_model_config({"extra": "allow"})


class Query(BaseModel):
    inputs: list[Item]
    """输入的内容"""

    refers: list[Item] = []
    """引用的内容"""

    # 定义配置
    model_config = default_model_config()


class Mention(BaseModel):
    uid: str
    """uid"""

    name: str
    """名称"""

    avatar: str | None = None
    """头像"""


class GenerateCmd(BaseModel):
    conversation_uid: str | None = None
    """会话ID"""

    query: Query
    """会话内容"""

    mentions: list[Mention] = []
    """@的机器人"""

    # 定义配置
    model_config = default_model_config()


async def clear_memory(current_user: Account, conversation_uid: str) -> list[Message]:
    """
    清除记忆
    :param current_user: 当前用户
    :param conversation_uid: 会话 ID
    """
    # TODO 这里需要考虑加锁

    # 找到会话
    conversation = await conversation_repository.get_by_uid(
        current_user.uid, conversation_uid
    )
    if not conversation:
        raise UnauthorizedError("找不到当前会话")

    # 找到最后一条消息
    latest_message = await message_repository.find_latest(
        conversation_uid, 1, reset_message_uid=conversation.reset_message_uid
    )
    latest_message = latest_message[-1] if latest_message else None

    if (
        not latest_message
        or latest_message.message_uid == conversation.reset_message_uid
    ):
        # 还没有消息，或者 最后一条消息就是重置消息
        # 防止重复重置
        return []

    # 保存消息
    reset_message = Message(
        conversation_uid=conversation_uid,
        sender_uid="system",
        sender_role="system",
        messages=[
            MessageBlock(
                type="system",
                content_type="text",
                content="以下为新对话",
                section_uid=BasePO.uid_generate(),
            )
        ],
        message_time=int(datetime.now().timestamp()) * 1000,
    )
    reset_message = await message_repository.add(reset_message)

    # 更新会话
    await conversation_repository.update_reset_message_uid(
        conversation_uid, reset_message.message_uid
    )

    # 清除消息总结
    await message_summary_repository.disable_all(conversation_uid)

    return [reset_message]


async def generate(
    current_user: Account, workspace_uid: str, chat_generate_cmd: GenerateCmd
) -> dict | ContentStream:
    """
    生成对话
    :param current_user: 当前用户
    :param workspace_uid: 空间ID
    :param chat_generate_cmd: 会话指令
    :return: 会话流
    """

    # 初始化 conversation
    first_text_message = next(
        filter(lambda item: item.type == "text", chat_generate_cmd.query.inputs)
    ).content
    conversation_name = (
        multi_replace(first_text_message, {"\n": " "})[:30]
        if first_text_message
        else "对话"
    )
    conversation = Conversation(
        conversation_uid=chat_generate_cmd.conversation_uid or "",
        creator_uid=current_user.uid,
        name=conversation_name,
    )
    conversation = await _init_generate_conversation(current_user, conversation)

    # 历史会话
    memory = ConversationSummaryBufferMemory(
        current_user=current_user,
        workspace_uid=workspace_uid,
        conversation=conversation,
    )
    summary_buffer_messages = await memory.get_summary_buffer_messages(
        to_langchain=True
    )
    summary_buffer_messages = ConversationSummaryBufferMemory.interleave_messages(
        summary_buffer_messages
    )

    # 存储 user message
    question_messages = (
        [  # @BOT
            MessageBlock(
                type="question",
                content_type="mention",
                content={
                    "uid": mention.uid,
                    "name": mention.name,
                    "avatar": mention.avatar,
                },
                section_uid=BasePO.uid_generate(),
            )
            for mention in chat_generate_cmd.mentions
        ]
        + [  # 引用
            MessageBlock(
                type="question",
                content_type=f"refer:{_refer.type}",
                content=_refer.content,
                section_uid=BasePO.uid_generate(),
            )
            for _refer in chat_generate_cmd.query.refers
        ]
        + [  # 消息
            MessageBlock(
                type="question",
                content_type=_input.type,
                content=_input.content,
                section_uid=BasePO.uid_generate(),
            )
            for _input in chat_generate_cmd.query.inputs
        ]
    )
    user_message = Message(
        conversation_uid=conversation.conversation_uid,
        sender_uid=current_user.uid,
        sender_role="user",
        messages=question_messages,
        message_time=int(datetime.now().timestamp()) * 1000,
    )
    await message_repository.add(user_message)

    # 编译机器人
    chat_bot = await _make_graph_bot(current_user, workspace_uid, chat_generate_cmd)

    # 机器人对话
    llm_event_stream = chat_bot.astream_events(
        {
            "messages": summary_buffer_messages
            + [HumanMessage(content=first_text_message)],
        },
        {"configurable": {"thread_id": "1"}},
        version="v2",
    )

    # 前置内容
    before_events = single_element_async_iterator(
        MessageStartEvent(section_uid=str(ULID()))
    )

    # 生成内容
    llm_message_events = llm_stream_events(llm_event_stream)

    # 后置内容
    after_events = single_element_async_iterator(
        MessageEndEvent(section_uid=str(ULID()))
    )

    # 合并内容
    message_events = merge_async_iterators(
        before_events,
        llm_message_events,
        after_events,
        yield_when_exception=lambda e: ErrorEvent(section_uid=str(ULID()), error=e),
    )

    return message_events_checkpoint(
        current_user,
        workspace_uid,
        message_events,
        conversation,
        chat_generate_cmd.mentions[0].uid if chat_generate_cmd.mentions else "",
    )


async def stop_generate(current_user: Account, conversation_uid: str) -> bool:
    """
    停止消息生成
    :param current_user: 当前用户
    :param conversation_uid: 会话 ID
    """

    # 找到会话
    conversation = await conversation_repository.get_by_uid(
        current_user.uid, conversation_uid
    )
    if not conversation:
        raise UnauthorizedError("找不到当前会话")

    # TODO 当前只支持单机local，需要考虑分布式集群部署的情况，将 stop 指令广播到所有进程进行处理
    generate_runner[conversation_uid] = False
    return True


async def _make_graph_bot(
    current_user: Account, workspace_uid: str, chat_generate_cmd: GenerateCmd
) -> CompiledGraph:
    """
    TODO 仅用于演示，需要重新设计
    """

    model_config = {}
    system_prompt = ""
    chat_model: BaseChatModel | None = None

    if chat_generate_cmd.mentions:
        bot = await bot_service.get_detail(chat_generate_cmd.mentions[0].uid)
        if not bot:
            raise UnauthorizedError("您无该智能体权限")
        if bot.mode != BotMode.SINGLE_AGENT:
            raise UnauthorizedError("暂仅支持单智能体")
        bot_config = bot.config or {}

        model_config = dict_get(bot_config, "model_config", {})
        # provider 配置
        provider_config = await llm_provider_service.detail(
            current_user, workspace_uid, dict_get(model_config, "provider_name")
        )
        provider_config.decrypt_credential()

        # provider → model → chat_model
        provider = model_provider_factory.get_provider(
            dict_get(model_config, "provider_name")
        )
        model: TextGenerationModel = provider.get_model(ModelType.TEXT_GENERATION)
        chat_model: BaseChatModel = (
            model.chat_model(
                provider_credential=provider_config.provider_credential,
                model_parameters=dict_get(model_config, "model_parameters"),
                model_name=dict_get(model_config, "model_name"),
                streaming=True,
                request_timeout=30,
                max_retries=3,
            )
            if model
            else None
        )

        system_prompt = dict_get(bot_config, "prompt_info.prompt", "")
    else:
        system_models = await llm_model_service.get_system_config(
            current_user, workspace_uid
        )
        if not system_models or ModelType.TEXT_GENERATION not in system_models:
            raise LLMExistsError(
                message="您还没有配置系统推理模型，请在 空间-设置-模型-系统模型设置 中添加 系统推理模型"
            )

        # model 配置
        model_config = system_models[ModelType.TEXT_GENERATION]

        # provider 配置
        provider_config = await llm_provider_service.detail(
            current_user, workspace_uid, model_config.provider_name
        )
        provider_config.decrypt_credential()

        # provider → model → chat_model
        provider = model_provider_factory.get_provider(model_config.provider_name)
        model: TextGenerationModel = provider.get_model(ModelType.TEXT_GENERATION)
        chat_model: BaseChatModel = (
            model.chat_model(
                provider_credential=provider_config.provider_credential,
                model_parameters=model_config.model_parameters,
                model_name=model_config.model_name,
                streaming=True,
                request_timeout=10,
                max_retries=3,
            )
            if model
            else None
        )

    if not chat_model:
        raise LLMExistsError(
            message=f"您在{model_config.provider_name}中还未配置任何{ModelType.TEXT_GENERATION}模型"
        )

    async def chat(state: MessagesState, config: RunnableConfig):
        system_message = SystemMessage(content=system_prompt or "")
        messages = state["messages"] or []
        if messages and isinstance(messages[0], SystemMessage):
            system_message = SystemMessage(
                content=f"{system_message.content}\n\n{messages[0].content}"
            )
            messages = messages[1:]
        invoke_messages, additional_kwargs = model.before_invoke([system_message] + messages, dict_get(model_config, "model_name"))
        response = await chat_model.ainvoke(
            invoke_messages, dict_merge(config, {"tags": ["explain_tag"]}, **additional_kwargs)
        )
        return {"messages": response}

    workflow = StateGraph(MessagesState)
    workflow.add_node("chat", chat)
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)

    return workflow.compile()


async def _init_generate_records(
    conversation: Conversation, user_message: Message
) -> tuple[str, str]:
    """
    初始化会话消息
    :param conversation: 当前用户
    :param user_message: 会话内容
    :return: (conversation_uid, message_uid)
    """

    conversation_uid = conversation.conversation_uid
    if not conversation_repository.get_by_uid(conversation_uid):
        _conversation = await conversation_repository.create(conversation)
        conversation_uid = _conversation.conversation_uid

    user_message.conversation_uid = conversation_uid
    await message_repository.add(user_message)

    # 生成新的 message_uid 给 assistant 使用
    return conversation_uid, BasePO.uid_generate()


async def _init_generate_conversation(
    current_user: Account, conversation: Conversation
) -> Conversation:
    """
    初始化会话
    :param conversation: 当前用户
    :return: Conversation
    """

    if conversation.conversation_uid:
        conversation = await conversation_repository.get_by_uid(
            current_user.uid, conversation.conversation_uid
        )

    if not conversation or not conversation.conversation_uid:
        conversation = await conversation_repository.create(conversation)

    return conversation


async def llm_stream_events(
    event_iter: AsyncIterator[StreamEvent],
) -> AsyncIterator[MessageEvent]:
    """
    通常，langgraph使用 BaseCheckpointSaver 处理历史消息
    langgraph.graph.state.StateGraph.compile(checkpoint=)
    https://langchain-ai.github.io/langgraph/how-tos/persistence/

    由于对历史消息的处理和端的展示高度耦合，同时对历史消息有定制化的诉求，langgraph的CheckpointSaver无法满足
    这里需要自定义处理逻辑

    :param event_iter: 异步事件流
    :return: 处理后的 异步事件流
    """

    section_uid = str(ULID())
    pre_type = None

    try:
        async for event in event_iter:
            logger.info(event)
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]

                # 判断当前类型
                if hasattr(chunk, "additional_kwargs") and ("reasoning_content" in chunk.additional_kwargs) and chunk.additional_kwargs["reasoning_content"]:
                    current_type = "reasoning_content"
                elif hasattr(chunk, "content") and chunk.content:
                    current_type = "content"
                else:
                    current_type = "other"
                    continue

                # 类型发生变化时需要重置section_uid，开启新的section
                if pre_type != current_type:
                    if pre_type == "reasoning_content":
                        yield AIReasoningChunkEvent(
                            section_uid=section_uid, reasoning=False, chunk=""
                        )

                    section_uid = str(ULID())
                    pre_type = current_type

                # 根据不同的类型返回不同的Event
                if current_type == "content":
                    yield AIMessageChunkEvent(
                        section_uid=section_uid, chunk=chunk.content
                    )
                elif current_type == "reasoning_content":
                    yield AIReasoningChunkEvent(
                        section_uid=section_uid, chunk=chunk.additional_kwargs["reasoning_content"]
                    )

                # refers 可能与 content 一起出现
                if hasattr(chunk, "additional_kwargs") and ("refers" in chunk.additional_kwargs) and chunk.additional_kwargs["refers"]:
                    section_uid = str(ULID())
                    yield AIReferCardsChunkEvent(
                        section_uid=section_uid, chunk=chunk.additional_kwargs["refers"]
                    )

            # TODO 其他类型判断
    except Exception as e:
        logger.exception(f"invoke llm stream error: {str(e)}")
        yield ErrorEvent(section_uid=str(ULID()), error=e)


async def message_events_checkpoint(
    current_user: Account,
    workspace_uid: str,
    event_iter: AsyncIterator[MessageEvent],
    conversation: Conversation,
    assistant_uid: str,
):
    checkpoint_saver = LLMMessageCheckPointSaver(
        current_user=current_user,
        workspace_uid=workspace_uid,
        conversation=conversation,
        assistant_uid=assistant_uid,
    )

    message_uid = BasePO.uid_generate()

    bot = await bot_service.get_detail(assistant_uid) if assistant_uid else None

    generate_runner[conversation.conversation_uid] = True
    async for event in event_iter:
        if not generate_runner.get(conversation.conversation_uid, True):
            # 用户停止了任务
            generate_runner.pop(conversation.conversation_uid, True)

            message = MessageEventData(
                conversation_uid=conversation.conversation_uid,
                sender_uid="system",
                sender_role="system",
                message_uid=BasePO.uid_generate(),
                message_time=int(datetime.now().timestamp()) * 1000,
                message=MessageBlockChunk(
                    type="system",
                    content_type="text",
                    content="用户终止了生成",
                    section_uid=BasePO.uid_generate(),
                    is_finished=True,
                ),
                is_finished=True,
            )
            checkpoint_saver.process(message)
            await checkpoint_saver.save()
            yield f"event: message\ndata: {message.model_dump_json()}\n\n"
            yield f"event: done\n\n"

            # FIXME 虽然不再向端推送，但是llm的请求并未终止，依然在持续生成（持续消耗token）
            return

        if isinstance(event, MessageStartEvent):
            message = MessageEventData(
                conversation_uid=conversation.conversation_uid,
                sender_uid=assistant_uid,
                sender_role="assistant",
                sender_info=(
                    SenderInfo(
                        uid=assistant_uid,
                        name=bot.name,
                        avatar=getattr(bot, "avatar_url", None),
                        role="assistant",
                    )
                    if bot
                    else None
                ),
                message_uid=message_uid,
                message_time=int(datetime.now().timestamp()) * 1000,
                message=MessageBlockChunk(
                    type="answer",
                    content_type="text",
                    content="",
                    section_uid=event.section_uid,
                ),
                is_finished=False,
            )
            yield f"event: message\ndata: {message.model_dump_json()}\n\n"

        if isinstance(event, AIMessageChunkEvent):
            message = MessageEventData(
                conversation_uid=conversation.conversation_uid,
                sender_uid=assistant_uid,
                sender_role="assistant",
                message_uid=message_uid,
                message_time=int(datetime.now().timestamp()) * 1000,
                message=MessageBlockChunk(
                    type="answer",
                    content_type="text",
                    content=event.chunk,
                    section_uid=event.section_uid,
                ),
                is_finished=False,
            )
            checkpoint_saver.process(message)
            yield f"event: message\ndata: {message.model_dump_json()}\n\n"

        if isinstance(event, AIReasoningChunkEvent):
            message = MessageEventData(
                conversation_uid=conversation.conversation_uid,
                sender_uid=assistant_uid,
                sender_role="assistant",
                message_uid=message_uid,
                message_time=int(datetime.now().timestamp()) * 1000,
                message=MessageBlockChunk(
                    type="answer",
                    content_type="reasoning:text",
                    content=event.chunk,
                    section_uid=event.section_uid,
                    is_finished=not event.reasoning,
                ),
                is_finished=False,
            )
            checkpoint_saver.process(message)
            # logger.debug(message)
            # logger.debug(event)
            yield f"event: message\ndata: {message.model_dump_json()}\n\n"

        if isinstance(event, AIReferCardsChunkEvent):
            message = MessageEventData(
                conversation_uid=conversation.conversation_uid,
                sender_uid=assistant_uid,
                sender_role="assistant",
                message_uid=message_uid,
                message_time=int(datetime.now().timestamp()) * 1000,
                message=MessageBlockChunk(
                    type="answer",
                    content_type="refer:cards",
                    content=event.chunk,
                    section_uid=event.section_uid,
                    is_finished=True,
                ),
                is_finished=False,
            )
            checkpoint_saver.process(message)
            yield f"event: message\ndata: {message.model_dump_json()}\n\n"

        if isinstance(event, MessageEndEvent):
            generate_runner.pop(conversation.conversation_uid, True)
            await checkpoint_saver.save()
            yield f"event: done\n\n"

        if isinstance(event, ErrorEvent):
            generate_runner.pop(conversation.conversation_uid, True)
            message = MessageEventData(
                conversation_uid=conversation.conversation_uid,
                sender_uid=assistant_uid,
                sender_role="assistant",
                message_uid=message_uid,
                message_time=int(datetime.now().timestamp()) * 1000,
                message=MessageBlockChunk(
                    type="answer",
                    content_type="error",
                    content=(
                        str(event.error)
                        if str(event.error)
                        else f"异常: {event.error.__class__.__name__}"
                    ),
                    section_uid=event.section_uid,
                    is_finished=True,
                ),
                is_finished=True,
            )
            checkpoint_saver.process(message)
            # await checkpoint_saver.save() -- 最后一定会走到MessageEndEvent，在MessageEndEvent中保存
            yield f"event: error\ndata: {message.model_dump_json()}\n\n"
