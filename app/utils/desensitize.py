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


def desensitize(secret: str, prefix: str = '') -> str:
    """
    脱敏
    :param secret: 待脱敏内容
    :param prefix: 脱敏内容追加前缀
    :return: 脱敏后的内容
    """
    length = len(secret)
    if length >= 12:
        return f"{prefix}{secret[:4]}{'*' * (length - 8)}{secret[-4:]}"
    elif 8 <= length < 12:
        return f"{prefix}{secret[:2]}{'*' * (length - 4)}{secret[-2:]}"
    elif 4 <= length < 8:
        return f"{prefix}{secret[:2]}{'*' * (length - 2)}"
    else:
        return f"{prefix}{'*' * length}"