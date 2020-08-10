#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import Counter
import logging as log
import numpy as np
import setting

# 一个分词器
# 读取包含古诗的文本，按规则过滤，生成词典
class Tokenizer:
	def __init__(self):
		# 按行存储古诗
		self.poetry = []
		# 词汇表
		self.tokens = None
		# 词->编号的映射
		self.token_2_id = None
		# 编号->词的映射
		self.id_2_token = None
		# 词汇表大小
		self.vocab_size = -1

	# 获取古诗列表，参数shuffle指定是否先混洗列表
	def get_poetry(self, shuffle:bool=True)->list:
		if shuffle:
			np.random.shuffle(self.poetry)
		return self.poetry

	# 读取文件，生成词典
	def gen_token_dict(self, filename: str)->None:
		self.__load_poetry(filename)
		self.tokens = self.__gen_tokens()
		self.token_2_id, self.id_2_token = self.__gen_token_dict()
		self.vocab_size = len(self.tokens)

	# 根据id返回token
	def id_to_token(self, token_id: int)->str:
		return self.id_2_token[token_id]

	# 给定一个词，查找它在词汇表中的编号
	def token_to_id(self, token:str)->int:
		return self.token_2_id.get(token, self.token_2_id['[UNK]'])

	# 给定一个字符串s，在头尾分别加上标记开始和结束的特殊字符，并将它转成对应的编号序列
	def encode(self, tokens:str)->list:
		# 加上开始标记
		token_ids = [self.token_to_id('[CLS]'), ]
		# 加入字符串编号序列
		for token in tokens:
			token_ids.append(self.token_to_id(token))
		# 加上结束标记
		token_ids.append(self.token_to_id('[SEP]'))
		return token_ids

	# 给定一个编号序列，将它解码成字符串
	def decode(self, token_ids:list)->str:
		# 起止标记字符特殊处理
		spec_tokens = {'[CLS]', '[SEP]'}
		# 保存解码出的字符的list
		tokens = []
		for token_id in token_ids:
			token = self.id_to_token(token_id)
			if token in spec_tokens:
				continue
			tokens.append(token)
		# 拼接字符串
		return ''.join(tokens)

	## 加载古诗文件
	def __load_poetry(self, filename: str) -> None:
		log.info("load poetry file `%s`...", filename)
		f = open(filename, 'r')
		# 每行一首古诗，格式为<题目>:<诗文>
		for line in f:
			# 将全角、半角的冒号统一替换成半角的
			line = line.replace('：', ':')
			# 提取题目和诗文
			title, content = line.split(":", 1)
			# 如果诗文中包含禁用词，则不使用这首诗
			if self.__has_diallowed_words(content):
				continue
			# 长度不能超过最大长度
			if len(content) > setting.MAX_LEN - 2:
				continue
			# 删除结尾的换行
			content = content.replace('\n', '')
			self.poetry.append(content)
		log.info("there are `%d` poetry", len(self.poetry))
		f.close()

	## 判断一首诗里面是否有停用词
	def __has_diallowed_words(self, content: str) -> bool:
		for c in content:
			if c in setting.DISALLOWED_WORDS:
				return True
		return False

	## 统计词频，生成token列表
	def __gen_tokens(self) -> list:
		counter = Counter()
		[counter.update(line) for line in self.poetry]
		tokens = [(tk, c) for tk, c in counter.items() if c >= setting.MIN_WORD_FREQUENCY]
		tokens = sorted(tokens, key = lambda x: -x[1])
		tokens = [tk for tk, __ in tokens]
		# 将特殊词和数据集中的词拼接起来
		return ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + tokens

	## 生成token和id的字典
	def __gen_token_dict(self) -> (dict, dict):
		tk2id = dict(zip(self.tokens, range(len(self.tokens))))
		id2tk = {value: key for key, value in tk2id.items()}
		return tk2id, id2tk

	