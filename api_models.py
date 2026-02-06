"""
MIT License

Copyright (c) Ollama

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from datetime import datetime
from typing import Any, Mapping, Optional

from pydantic import BaseModel, ByteSize, Field


# https://github.com/ollama/ollama-python/blob/0008226fda83c7b0c6722844313a5adfae30001c/ollama/_types.py#L19-L101
class SubscriptableBaseModel(BaseModel):  # non-public
    def __getitem__(self, key: str) -> Any:
        """
        >>> msg = Message(role='user')
        >>> msg['role']
        'user'
        >>> msg = Message(role='user')
        >>> msg['nonexistent']
        Traceback (most recent call last):
        KeyError: 'nonexistent'
        """
        if key in self:  # pyright: ignore[reportOperatorIssue]
            return getattr(self, key)

        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        >>> msg = Message(role='user')
        >>> msg['role'] = 'assistant'
        >>> msg['role']
        'assistant'
        >>> tool_call = Message.ToolCall(function=Message.ToolCall.Function(name='foo', arguments={}))
        >>> msg = Message(role='user', content='hello')
        >>> msg['tool_calls'] = [tool_call]
        >>> msg['tool_calls'][0]['function']['name']
        'foo'
        """
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """
        >>> msg = Message(role='user')
        >>> 'nonexistent' in msg
        False
        >>> 'role' in msg
        True
        >>> 'content' in msg
        False
        >>> msg.content = 'hello!'
        >>> 'content' in msg
        True
        >>> msg = Message(role='user', content='hello!')
        >>> 'content' in msg
        True
        >>> 'tool_calls' in msg
        False
        >>> msg['tool_calls'] = []
        >>> 'tool_calls' in msg
        True
        >>> msg['tool_calls'] = [Message.ToolCall(function=Message.ToolCall.Function(name='foo', arguments={}))]
        >>> 'tool_calls' in msg
        True
        >>> msg['tool_calls'] = None
        >>> 'tool_calls' in msg
        True
        >>> tool = Tool()
        >>> 'type' in tool
        True
        """
        if key in self.model_fields_set:
            return True

        if value := self.__class__.model_fields.get(key):
            return value.default is not None

        return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        >>> msg = Message(role='user')
        >>> msg.get('role')
        'user'
        >>> msg = Message(role='user')
        >>> msg.get('nonexistent')
        >>> msg = Message(role='user')
        >>> msg.get('nonexistent', 'default')
        'default'
        >>> msg = Message(role='user', tool_calls=[ Message.ToolCall(function=Message.ToolCall.Function(name='foo', arguments={}))])
        >>> msg.get('tool_calls')[0]['function']['name']
        'foo'
        """
        return getattr(self, key) if hasattr(self, key) else default


# vvvvvvvv All None has been replaced with default value vvvvvvvv


# https://github.com/ollama/ollama-python/blob/0008226fda83c7b0c6722844313a5adfae30001c/ollama/_types.py#L488-L494
class ModelDetails(SubscriptableBaseModel):  # non-public
    parent_model: str = ""
    format: str = ""
    family: str = ""
    families: list[str] = [""]
    parameter_size: str = ""
    quantization_level: str = ""


# https://github.com/ollama/ollama-python/blob/0008226fda83c7b0c6722844313a5adfae30001c/ollama/_types.py#L497-L505
class ListResponse(SubscriptableBaseModel):
    class Model(SubscriptableBaseModel):
        name: str  # new field
        model: str = ""
        modified_at: datetime | str = ""
        digest: str = ""
        size: ByteSize
        details: ModelDetails

    models: list[Model]


# https://github.com/ollama/ollama-python/blob/0008226fda83c7b0c6722844313a5adfae30001c/ollama/_types.py#L543-L558
class ShowResponse(SubscriptableBaseModel):
    modified_at: datetime | str = ""
    template: str = ""
    modelfile: str = ""
    license: str = ""
    details: ModelDetails
    modelinfo: Mapping[str, Any] = Field(alias="model_info", default={})
    parameters: str = ""
    capabilities: list[str] = [""]


# ^^^^^^^^  All None has been replaced with default value ^^^^^^^^


# https://docs.ollama.com/api-reference/show-model-details
class ShowRequest(BaseModel):
    model: str
    verbose: Optional[bool] = False


# https://docs.ollama.com/api-reference/get-version
class VersionResponse(BaseModel):
    version: str
