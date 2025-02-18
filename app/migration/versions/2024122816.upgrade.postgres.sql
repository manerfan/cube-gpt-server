-- 创建uuid扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
-- 创建pgcrypto扩展
CREATE EXTENSION IF NOT EXISTS pgcrypto;
-- 创建vector扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 更行数据时更新updated_at
CREATE FUNCTION update_updated_at_column() RETURNS trigger LANGUAGE plpgsql AS
$$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        NEW.updated_at = CURRENT_TIMESTAMP(0);
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$;

-- 用户
CREATE TABLE modu_accounts
(
    id          UUID                     DEFAULT UUID_GENERATE_V4()          NOT NULL CONSTRAINT pk_modu_account_id PRIMARY KEY,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0)        NOT NULL,
    updated_at  TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0)        NOT NULL,
    uid         VARCHAR(32)                                                  NOT NULL,
    name        VARCHAR(128)                                                 NOT NULL,
    email       VARCHAR(128)                                                 NOT NULL,
    password    VARCHAR(128)                                                 NOT NULL,
    avatar      VARCHAR(256),
    description VARCHAR(512),
    status      VARCHAR(32)              DEFAULT 'active'::CHARACTER VARYING NOT NULL,
    is_deleted  SMALLINT                 DEFAULT '0'::SMALLINT               NOT NULL
);
CREATE UNIQUE INDEX uk_modu_account_uid ON modu_accounts (uid);
CREATE UNIQUE INDEX uk_modu_account_email ON modu_accounts (email);
CREATE TRIGGER set_modu_accounts_updated_at
    BEFORE UPDATE
    ON modu_accounts
    FOR EACH ROW
EXECUTE procedure update_updated_at_column();

-- 工作区
CREATE TABLE modu_workspaces
(
    id          UUID                     DEFAULT UUID_GENERATE_V4()   NOT NULL CONSTRAINT pk_modu_workspace_id PRIMARY KEY,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0) NOT NULL,
    updated_at  TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0) NOT NULL,
    uid         VARCHAR(32)                                           NOT NULL,
    creator_uid VARCHAR(32)                                           NOT NULL,
    name        VARCHAR(128)                                          NOT NULL,
    description VARCHAR(512),
    type        VARCHAR(32)                                           NOT NULL,
    is_deleted  SMALLINT                 DEFAULT '0'::SMALLINT        NOT NULL
);
CREATE UNIQUE INDEX uk_modu_workspace_uid ON modu_workspaces (uid);
CREATE TRIGGER set_modu_workspaces_updated_at
    BEFORE UPDATE
    ON modu_workspaces
    FOR EACH ROW
EXECUTE procedure update_updated_at_column();

-- 工作区成员
CREATE TABLE modu_workspace_membership
(
    id            UUID                     DEFAULT UUID_GENERATE_V4()           NOT NULL CONSTRAINT pk_modu_workspace_membership_id PRIMARY KEY,
    created_at    TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0)         NOT NULL,
    updated_at    TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0)         NOT NULL,
    uid           VARCHAR(32)                                                   NOT NULL,
    workspace_uid VARCHAR(32)                                                   NOT NULL,
    member_uid    VARCHAR(32)                                                   NOT NULL,
    member_role   VARCHAR(32)                                                   NOT NULL,
    member_status VARCHAR(32)              DEFAULT 'PENDING'::CHARACTER VARYING NOT NULL,
    is_deleted  SMALLINT                   DEFAULT '0'::SMALLINT        NOT NULL
);
CREATE UNIQUE INDEX uk_modu_workspace_membership_uid ON modu_workspace_membership (uid);
CREATE INDEX idx_modu_workspace_membership_workspace_uid ON modu_workspace_membership (workspace_uid, member_uid, member_status);
CREATE INDEX idx_modu_workspace_membership_member_uid ON modu_workspace_membership (member_uid, member_status, workspace_uid);
CREATE TRIGGER set_modu_workspace_membership_updated_at
    BEFORE UPDATE
    ON modu_workspace_membership
    FOR EACH ROW
EXECUTE procedure update_updated_at_column();

-- LLM Provider 配置
CREATE TABLE modu_llm_provider_config
(
    id            UUID                     DEFAULT UUID_GENERATE_V4()   NOT NULL CONSTRAINT pk_modu_llm_provider_config_id PRIMARY KEY,
    created_at    TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0) NOT NULL,
    updated_at    TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0) NOT NULL,
    uid           VARCHAR(32)                                           NOT NULL,
    workspace_uid VARCHAR(32)                                           NOT NULL,
    provider_name  VARCHAR(32)                                           NOT NULL,
    provider_credential VARCHAR                                         NOT NULL,
    is_deleted  SMALLINT                   DEFAULT '0'::SMALLINT        NOT NULL
);
CREATE UNIQUE INDEX uk_modu_llm_provider_uid ON modu_llm_provider_config (uid);
CREATE UNIQUE INDEX uk_modu_llm_provider_space_key ON modu_llm_provider_config (workspace_uid, provider_name);
CREATE TRIGGER set_modu_llm_provider_updated_at
    BEFORE UPDATE
    ON modu_llm_provider_config
    FOR EACH ROW
EXECUTE procedure update_updated_at_column();

-- LLM 系统默认模型配置
CREATE TABLE modu_llm_system_model_config
(
    id            UUID                     DEFAULT UUID_GENERATE_V4()   NOT NULL CONSTRAINT pk_modu_llm_system_model_config_id PRIMARY KEY,
    created_at    TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0) NOT NULL,
    updated_at    TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0) NOT NULL,
    uid           VARCHAR(32)                                           NOT NULL,
    workspace_uid VARCHAR(32)                                           NOT NULL,
    model_config  JSONB                                                 NOT NULL,
    is_deleted  SMALLINT                   DEFAULT '0'::SMALLINT        NOT NULL
);
CREATE UNIQUE INDEX uk_modu_llm_system_model_config_uid ON modu_llm_system_model_config (uid);
CREATE UNIQUE INDEX uk_modu_llm_system_model_config_space ON modu_llm_system_model_config (workspace_uid);
CREATE TRIGGER set_modu_llm_system_model_config_updated_at
    BEFORE UPDATE
    ON modu_llm_system_model_config
    FOR EACH ROW
EXECUTE procedure update_updated_at_column();

-- 会话
CREATE TABLE modu_conversations
(
    id                  UUID                     DEFAULT UUID_GENERATE_V4()             NOT NULL CONSTRAINT pk_modu_conversations_id PRIMARY KEY,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0)           NOT NULL,
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0)           NOT NULL,
    uid                 VARCHAR(32)                                                     NOT NULL,
    creator_uid         VARCHAR(32)                                                     NOT NULL,
    workspace_uid       VARCHAR(32)                                                     NOT NULL,
    name                VARCHAR(64)                                                     NOT NULL,
    reset_message_uid   VARCHAR(64)                                                     ,
    scope               VARCHAR(32)              DEFAULT 'GLOBAL'::CHARACTER VARYING    NOT NULL,
    is_deleted          SMALLINT                 DEFAULT '0'::SMALLINT                  NOT NULL
);
CREATE UNIQUE INDEX uk_modu_conversations_uid ON modu_conversations (uid);
CREATE INDEX idx_modu_conversations_creator ON modu_conversations (scope, creator_uid, uid);
CREATE TRIGGER set_modu_conversations_updated_at
    BEFORE UPDATE
    ON modu_conversations
    FOR EACH ROW
EXECUTE procedure update_updated_at_column();

-- 消息
CREATE TABLE modu_messages
(
    id                  UUID                     DEFAULT UUID_GENERATE_V4()     NOT NULL CONSTRAINT pk_modu_messages_id PRIMARY KEY,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0)   NOT NULL,
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0)   NOT NULL,
    uid                 VARCHAR(32)                                             NOT NULL,
    conversation_uid    VARCHAR(32)                                             NOT NULL,
    message_time        BIGINT                                                  NOT NULL,
    sender_uid          VARCHAR(32)                                             NOT NULL,
    sender_role         VARCHAR(32)                                             NOT NULL,
    messages            JSONB                                                   NOT NULL,
    is_deleted          SMALLINT                 DEFAULT '0'::SMALLINT          NOT NULL
);
CREATE UNIQUE INDEX uk_modu_messages_uid ON modu_messages (uid);
CREATE INDEX idx_modu_messages_conversation ON modu_messages (conversation_uid);
CREATE TRIGGER set_modu_messages_updated_at
    BEFORE UPDATE
    ON modu_messages
    FOR EACH ROW
EXECUTE procedure update_updated_at_column();

-- 消息总结
CREATE TABLE modu_message_summary
(
    id                  UUID                     DEFAULT UUID_GENERATE_V4()     NOT NULL CONSTRAINT pk_modu_message_summary_id PRIMARY KEY,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0)   NOT NULL,
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0)   NOT NULL,
    uid                 VARCHAR(32)                                             NOT NULL,
    conversation_uid    VARCHAR(32)                                             NOT NULL,
    summary             TEXT                                                    NOT NULL,
    summary_order       BIGINT                   DEFAULT '0'::BIGINT            NOT NULL,
    last_message_uid    VARCHAR(32)                                             NOT NULL,
    is_deleted          SMALLINT                 DEFAULT '0'::SMALLINT          NOT NULL
);
CREATE UNIQUE INDEX uk_modu_message_summary_uid ON modu_message_summary (uid);
CREATE UNIQUE INDEX uk_modu_message_summary_conversation_order ON modu_message_summary (conversation_uid, summary_order);
CREATE INDEX idx_modu_message_summary_conversation ON modu_message_summary (conversation_uid, last_message_uid, summary_order);
CREATE TRIGGER set_modu_message_summary_updated_at
    BEFORE UPDATE
    ON modu_message_summary
    FOR EACH ROW
EXECUTE procedure update_updated_at_column();

-- 智能体/机器人
CREATE TABLE modu_bots
(
    id                  UUID                     DEFAULT UUID_GENERATE_V4()                 NOT NULL CONSTRAINT pk_modu_bot_id PRIMARY KEY,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0)               NOT NULL,
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0)               NOT NULL,
    uid                 VARCHAR(32)                                                         NOT NULL,
    workspace_uid       VARCHAR(32)                                                         NOT NULL,
    name                VARCHAR(128)                                                        NOT NULL,
    avatar              VARCHAR(256)                                                               ,
    description         TEXT                                                                       ,
    creator_uid         VARCHAR(32)                                                         NOT NULL,
    mode                VARCHAR(32)              DEFAULT 'SINGLE_AGENT'::CHARACTER VARYING  NOT NULL,
    config              JSONB                                                                      ,
    publish_uid         VARCHAR(32)                                                                ,
    is_deleted          SMALLINT                 DEFAULT '0'::SMALLINT                      NOT NULL
);
CREATE UNIQUE INDEX uk_modu_bot_uid ON modu_bots (uid);
CREATE INDEX idx_modu_bot_space ON modu_bots (workspace_uid, is_deleted);
CREATE INDEX idx_modu_bot_publish_uid ON modu_bots (publish_uid, is_deleted);
CREATE TRIGGER set_modu_bots_updated_at
    BEFORE UPDATE
    ON modu_bots
    FOR EACH ROW
EXECUTE procedure update_updated_at_column();

-- 配置发布
CREATE TABLE modu_publish_configs
(
    id                  UUID                     DEFAULT UUID_GENERATE_V4()         NOT NULL CONSTRAINT pk_modu_publish_config_id PRIMARY KEY,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0)       NOT NULL,
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0)       NOT NULL,
    uid                 VARCHAR(32)                                                 NOT NULL,
    target_type         VARCHAR(32)                                                 NOT NULL,
    target_uid          VARCHAR(32)                                                 NOT NULL,
    config              JSONB                                                       NOT NULL,
    creator_uid         VARCHAR(32)                                                 NOT NULL,
    publish_status      VARCHAR(32)              DEFAULT 'DRAFT'::CHARACTER VARYING NOT NULL,
    is_deleted          SMALLINT                 DEFAULT '0'::SMALLINT              NOT NULL
);
CREATE UNIQUE INDEX uk_modu_publish_config_uid ON modu_publish_configs (uid);
CREATE INDEX idx_modu_publish_config_target ON modu_publish_configs (target_type, target_uid, publish_status, is_deleted);
CREATE TRIGGER set_modu_publish_configs_updated_at
    BEFORE UPDATE
    ON modu_publish_configs
    FOR EACH ROW
EXECUTE procedure update_updated_at_column();

-- 收藏
CREATE TABLE modu_favorites
(
    id                  UUID                     DEFAULT UUID_GENERATE_V4()         NOT NULL CONSTRAINT pk_modu_favorite_id PRIMARY KEY,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0)       NOT NULL,
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0)       NOT NULL,
    uid                 VARCHAR(32)                                                 NOT NULL,
    target_type         VARCHAR(32)                                                 NOT NULL,
    target_uid          VARCHAR(32)                                                 NOT NULL,
    creator_uid         VARCHAR(32)                                                 NOT NULL,
    is_deleted          SMALLINT                 DEFAULT '0'::SMALLINT              NOT NULL
);
CREATE UNIQUE INDEX uk_modu_favorite_uid ON modu_favorites (uid);
CREATE UNIQUE INDEX uk_modu_favorite_target ON modu_favorites (target_type, target_uid, creator_uid, is_deleted);
CREATE INDEX idx_modu_favorite_creator ON modu_favorites (creator_uid, target_type, is_deleted);
CREATE TRIGGER set_modu_favorites_updated_at
    BEFORE UPDATE
    ON modu_favorites
    FOR EACH ROW
EXECUTE procedure update_updated_at_column();

-- 文件
CREATE TABLE modu_files
(
    id                  UUID                     DEFAULT UUID_GENERATE_V4()     NOT NULL CONSTRAINT pk_modu_file_id PRIMARY KEY,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0)   NOT NULL,
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP(0)   NOT NULL,
    uid                 VARCHAR(64)                                             NOT NULL,
    file_key            VARCHAR(64)                                             NOT NULL,
    filename            VARCHAR(256)                                            NOT NULL,
    file_mime_type      VARCHAR(128)                                            NOT NULL,
    file_category       VARCHAR(64)                                             NOT NULL,
    file_size           BIGINT                                                  NOT NULL,
    file_hash           VARCHAR(256)                                            NOT NULL,
    storage_type        VARCHAR(256)                                            NOT NULL,
    creator_uid         VARCHAR(32)                                             NOT NULL,
    is_deleted          SMALLINT                 DEFAULT '0'::SMALLINT          NOT NULL
);
CREATE UNIQUE INDEX uk_modu_file_uid ON modu_files (uid);
CREATE UNIQUE INDEX uk_modu_file_key ON modu_files (file_key);
CREATE INDEX idx_modu_file_storage ON modu_files (file_key, storage_type, is_deleted);
CREATE TRIGGER set_modu_files_updated_at
    BEFORE UPDATE
    ON modu_files
    FOR EACH ROW
EXECUTE procedure update_updated_at_column();
