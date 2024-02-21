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

from sqlalchemy.orm import Session

from .account_models import Account
from ..database import Repository, Database


class AccountRepository(Repository):
    """
    账号数据存储的定义
    """

    def __init__(self, database: Database):
        Repository.__init__(self, database)

    @abstractmethod
    def find_one_by_email(self, email: str, session: Session) -> Account:
        """
        通过email查找账号
        :param email: 邮箱
        :param session: Session
        :return: 账号
        """
        raise NotImplementedError()

    @abstractmethod
    def create(self, account: Account, session: Session) -> Account:
        """
        创建账号
        :param account: 账号
        :param session: Session
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def count_all(self, session: Session) -> int:
        """
        统计所有账号
        :param session: Session
        :return: 账号数量
        """
        raise NotImplementedError()
