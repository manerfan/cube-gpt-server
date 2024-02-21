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

from sqlalchemy import PrimaryKeyConstraint, String, Enum, text
from sqlalchemy.orm import Mapped, mapped_column, Session

from repositories.data.database_postgres import PostgresBasePO
from utils.errors.account_error import AccountLoginError
from .AccountRepository import AccountRepository
from .account_models import Account, AccountStatus
from ..database import with_session, BasePO


class AccountRepositoryPostgres(AccountRepository):
    """
    账号数据存储的PostgreSQL实现
    """

    @with_session
    def find_one_by_email(self, email: str, session: Session) -> Account:
        account_model = session.query(AccountPO).filter(AccountPO.email == email).first()
        if not account_model:
            raise AccountLoginError(message='邮箱或密码错误')
        return Account(**account_model.as_dict())

    @with_session
    def create(self, account: Account, session: Session) -> Account:
        account_po = AccountPO(**account.__dict__)
        account_po.uid = BasePO.uid_generate()
        session.add(account_po)

        account.uid = account_po.uid
        return account

    @with_session
    def count_all(self, session: Session) -> int:
        return session.query(AccountPO).count()


class AccountPO(PostgresBasePO):
    """
    账号PO
    """

    __tablename__ = 'cube_accounts'
    __table_args__ = (
        PrimaryKeyConstraint('id', name='pk_id'),
    )

    name: Mapped[str] = mapped_column(String(128), nullable=False, comment='用户名')
    email: Mapped[str] = mapped_column(String(128), nullable=False, comment='邮箱')
    password: Mapped[str] = mapped_column(String(128), nullable=False, comment='密码')
    avatar: Mapped[str] = mapped_column(String(256), nullable=True, comment='头像')
    status: Mapped[AccountStatus] = mapped_column(Enum(AccountStatus), nullable=False,
                                                  server_default=text("'active'::character varying"),
                                                  comment='账号状态')
