# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
核心方法为PromptEncoder:
    1. 先定义一个virtual_token序列的长度,
        计算最终的virtual_token序列的长度为config.num_virtual_tokens * config.num_transformer_submodules；
        意味着每个transformer子模块都有独立的virtual_token;
    2. 然后定义virtual_token索引的embedding层；
    3. 先加载指定的tokenizer;
    4. 对初始文本进行tokenize,记录初始文本tokenizer后的长度；
    5. 如果初始文本tokenizer后的长度大于虚拟token的长度，则初始文本内容只取虚拟token的长度；
        如果初始文本tokenizer后的长度小于虚拟token的长度，则将初始文本内容重复填充至虚拟token的长度；
        从而确保初始文本tokenizer后的长度等于虚拟token的长度；
    6. 使用基础模型的词嵌入层对初始文本tokenized进行嵌入，并转换为float32位；
    7. 将基础模型嵌入并转换后的权重设为可训练参数，并赋值给virtual_token的嵌入层的权重，亦即以prompt初始化文本的嵌入向量为初始权重
    8. 训练时只需获取给定虚拟token对应的嵌入向量。
"""
import enum
import math
from dataclasses import dataclass, field
from typing import Optional, Union

import torch

from ..utils import PeftType, PromptLearningConfig


class PromptTuningInit(str, enum.Enum):
    TEXT = "TEXT"
    RANDOM = "RANDOM"


@dataclass
class PromptTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`PromptEmbedding`].

    Args:
        prompt_tuning_init (Union[[`PromptTuningInit`], `str`]): The initialization of the prompt embedding.
        prompt_tuning_init_text (`str`, *optional*):
            The text to initialize the prompt embedding. Only used if `prompt_tuning_init` is `TEXT`.
        tokenizer_name_or_path (`str`, *optional*):
            The name or path of the tokenizer. Only used if `prompt_tuning_init` is `TEXT`.
    """

    prompt_tuning_init: Union[PromptTuningInit, str] = field(
        default=PromptTuningInit.RANDOM,
        metadata={"help": "How to initialize the prompt tuning parameters"},
    )
    prompt_tuning_init_text: Optional[str] = field(
        default=None,
        metadata={
            "help": "The text to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.PROMPT_TUNING


class PromptEmbedding(torch.nn.Module):
    """
    The model to encode virtual tokens into prompt embeddings.

    Args:
        config ([`PromptTuningConfig`]): The configuration of the prompt embedding.
        word_embeddings (`torch.nn.Module`): The word embeddings of the base transformer model.

    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) -- The embedding layer of the prompt embedding.

    Example:

    ```py
    >>> from peft import PromptEmbedding, PromptTuningConfig

    >>> config = PromptTuningConfig(
    ...     peft_type="PROMPT_TUNING",
    ...     task_type="SEQ_2_SEQ_LM",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     num_transformer_submodules=1,
    ...     num_attention_heads=12,
    ...     num_layers=12,
    ...     prompt_tuning_init="TEXT",
    ...     prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral",
    ...     tokenizer_name_or_path="t5-base",
    ... )

    >>> # t5_model.shared is the word embeddings of the base model
    >>> prompt_embedding = PromptEmbedding(config, t5_model.shared)
    ```

    Input Shape: (`batch_size`, `total_virtual_tokens`)

    Output Shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
    """

    def __init__(self, config, word_embeddings):
        super().__init__()

        total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
        self.embedding = torch.nn.Embedding(total_virtual_tokens, config.token_dim)
        # PromptTuning的初始化有两种TEXT，RANDOM;
        if config.prompt_tuning_init == PromptTuningInit.TEXT:
            # 1. 先加载指定的tokenizer;
            # 2. 对初始文本进行tokenize,记录初始文本tokenizer后的长度；
            # 3. 如果初始文本tokenizer后的长度大于虚拟token的长度，则初始文本内容只取虚拟token的长度；
            # 4. 如果初始文本tokenizer后的长度小于虚拟token的长度，则将初始文本内容重复填充至虚拟token的长度；
            # 5. 从而确保初始文本tokenizer后的长度等于虚拟token的长度；
            # 6. 使用基础模型的词嵌入层对初始文本tokenized进行嵌入，并转换为float32位；
            # 7. 将基础模型嵌入并转换后的权重设为可训练参数，并赋值给virtual_token的嵌入层的权重；
            # 8. 训练时只需获取给定虚拟token对应的嵌入向量。
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
            init_text = config.prompt_tuning_init_text
            init_token_ids = tokenizer(init_text)["input_ids"]
            # Trim or iterate until num_text_tokens matches total_virtual_tokens
            num_text_tokens = len(init_token_ids)
            if num_text_tokens > total_virtual_tokens:
                init_token_ids = init_token_ids[:total_virtual_tokens]
            elif num_text_tokens < total_virtual_tokens:
                num_reps = math.ceil(total_virtual_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps
            init_token_ids = init_token_ids[:total_virtual_tokens]
            #  The word embeddings of the base transformer model
            word_embedding_weights = word_embeddings(torch.LongTensor(init_token_ids)).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.embedding.weight = torch.nn.Parameter(word_embedding_weights)

    def forward(self, indices):
        # Just get embeddings
        prompt_embeddings = self.embedding(indices)
        return prompt_embeddings
