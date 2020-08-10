#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

# 导入自定义模块
from generator import PoetryDataGenerator
import tokenizer
import setting

# 随机生成一首诗
#  - tokenizer: 分词器
#  - model: 用于生成古诗的模型
#  - s: 用于生成古诗的起始字符串，默认为空串
# return: 一个字符串，表示一首古诗
def generate_random_poetry(tokenizer, model, s=''):
	# 将初始字符串转成token
	token_ids = tokenizer.encode(s)
	# 去掉结束标记[SEP]
	token_ids = token_ids[:-1]
	while len(token_ids) < settings.MAX_LEN:
		# 进行预测，只保留第一个样例（我们输入的样例数只有1）的、最后一个token的预测的、不包含[PAD][UNK][CLS]的概率分布
		_probas = model.predict([token_ids, ])[0, -1, 3:]
		# print(_probas)
		# 按照出现概率，对所有token倒序排列
		p_args = _probas.argsort()[::-1][:100]
		# 排列后的概率顺序
		p = _probas[p_args]
		# 先对概率归一
		p = p / sum(p)
		# 再按照预测出的概率，随机选择一个词作为预测结果
		target_index = np.random.choice(len(p), p=p)
		target = p_args[target_index] + 3
		# 保存
		token_ids.append(target)
		if target == 3:
			break
	return tokenizer.decode(token_ids)

# 在每个epoch训练完成后，保留最优权重，并随机生成settings.SHOW_NUM首古诗展示
class Evaluate(tf.keras.callbacks.Callback):
	def __init__(self, tokenizer):
		super().__init__()
		# 给loss赋一个较大的初始值
		self.lowest = 1e10

	def on_epoch_end(self, epoch, logs=None):
		# 在每个epoch训练完成后调用
		# 如果当前loss更低，就保存当前模型参数
		if logs['loss'] <= self.lowest:
			self.lowest = logs['loss']
			model.save(setting.BEST_MODEL_PATH)
		# 随机生成几首古体诗测试，查看训练效果
		print()
		for i in range(setting.SHOW_NUM):
			print(utils.generate_random_poetry(tokenizer, model))