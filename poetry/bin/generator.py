#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import math

# 古诗数据集生成器
class PoetryDataGenerator:
	def __init__(self, data, batch_size, tokenizer, random=False):
		# 数据集
		self.data = data
		# batch size
		self.batch_size = batch_size
		# 每个epoch迭代的步数
		self.steps = int(math.floor(len(self.data)/self.batch_size))
		# 每个epoch开始时是否随机混洗
		self.random = random
		# 分词器
		self.tokenizer = tokenizer

	# 将给定数据填充到相同长度
	#  - length: 填充后的长度，不传递此参数则使用data中的最大长度
	#  - padding: 用于填充的数据，不传递此参数则使用[PAD]的对应编号
	def sequence_padding(self, data, length=None, padding=None):
		# 计算填充长度
		if length is None:
			length = max(map(len, data))
		# 计算填充数据
		if padding is None:
			padding = self.tokenizer.token_to_id('[PAD]')
		# 开始填充
		outputs = []
		for line in data:
			padding_length = length - len(line)
			# 不足就进行填充
			if padding_length > 0:
				outputs.append(np.concatenate([line, [padding] * padding_length]))
			# 超过就进行截断
			else:
				outputs.append(line[:length])
		return np.array(outputs)

	def __len__(self):
		return self.steps

	def __iter__(self):
		total = len(self.data)
		# 是否随机混洗
		if self.random:
			np.random.shuffle(self.data)
		# 迭代一个epoch，每次yield一个batch
		for start in range(0, total, self.batch_size):
			end = min(start + self.batch_size, total)
			batch_data = []
			# 逐一对古诗进行编码
			for single_data in self.data[start:end]:
				batch_data.append(self.tokenizer.encode(single_data))
			# 填充为相同长度
			batch_data = self.sequence_padding(batch_data)
			# yield x,y
			yield batch_data[:, :-1], tf.one_hot(batch_data[:, 1:], self.tokenizer.vocab_size)
			del batch_data

	# 创建一个生成器，用于训练
	def for_fit(self):
		# 死循环，当数据训练一个epoch之后，重新迭代数据
		while True:
			# 委托生成器
			yield from self.__iter__()