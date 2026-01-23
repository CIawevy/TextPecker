"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import io
import json
import sys
import random

import numpy as np

from synthtiger import utils
from synthtiger.components.component import Component


class EnglishCorpus(Component):
    def __init__(
        self,
        paths=(),
        weights=(),
        min_length=None,
        max_length=None,
    ):
        super().__init__()
        self.paths = paths
        self.weights = weights
        self.min_length = min_length
        self.max_length = max_length
        self._contents = []  # 存储所有文档的内容
        self._counts = []    # 存储每个文件中的文档数量
        self._probs = np.array(self.weights) / sum(self.weights) if weights else None
        self._update_contents()

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        if len(self.paths) == 0:
            raise RuntimeError("Corpus path is not specified")
        if len(self.paths) != len(self.weights):
            raise RuntimeError(
                "The number of weights does not match the number of corpus paths"
            )

        text = self._sample_text()
        text = meta.get("text", text)

        meta = {
            "text": text,
        }

        return meta

    def data(self, meta):
        text = meta["text"]
        return text

    def _update_contents(self):
        """更新内容，读取JSONL文件并存储content字段"""
        self._contents = []
        self._counts = []

        for path in self.paths:
            documents = []
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    for line in fp:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            doc = json.loads(line)
                            # 提取content字段
                            if "content" in doc:
                                content = doc["content"]
                                # 预处理内容：移除多余的换行符，替换为空格
                                content = content.replace('\n', ' ')
                                # 合并多个空格为一个
                                content = ' '.join(content.split())
                                documents.append(content)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Error reading file {path}: {e}")
                documents = []

            self._contents.append(documents)
            self._counts.append(len(documents))

    def _sample_text(self):
        """采样文本，根据配置的单词数量范围"""
        # 随机选择一个文件
        key = np.random.choice(len(self.paths), p=self._probs)
        if self._counts[key] == 0:
            raise RuntimeError(f"There is no text: {self.paths[key]}")

        # 随机选择一个文档
        doc_idx = np.random.randint(self._counts[key])
        document = self._contents[key][doc_idx]

        # 如果没有指定长度限制，直接返回整个文档
        if self.min_length is None and self.max_length is None:
            return document

        # 将文档按空格分割成单词
        words = document.split()
        if not words:
            return ""

        # 确定要采样的单词数量
        min_words = self.min_length if self.min_length is not None else 1
        max_words = self.max_length if self.max_length is not None else len(words)
        # 确保max_words不大于文档中的单词总数
        max_words = min(max_words, len(words))
        # 如果min_words大于max_words，则调整
        if min_words > max_words:
            min_words = max_words

        # 随机确定要采样的单词数量
        num_words = random.randint(min_words, max_words)
        
        # 随机确定起始位置
        if len(words) == num_words:
            # 如果文档单词数刚好等于要采样的数量，取全部
            start_idx = 0
        else:
            # 否则随机选择起始位置
            start_idx = random.randint(0, len(words) - num_words)
        
        # 提取连续的单词序列
        sampled_words = words[start_idx:start_idx + num_words]
        # 组合成文本
        sampled_text = ' '.join(sampled_words)
        
        return sampled_text