#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import Counter
from krait import config
import tensorflow as tf
import logging as log
import pandas as pd
import argparse
import json
import sys
import os

# 引入自定义模块
from generator import PoetryDataGenerator
from evaluate import Evaluate
import tokenizer
import setting

# 配置命令行参数
__ARGS_INFO__ = """
# 应用程序名
"application" = "poetry::app"

# 应用级参数列表
"application_args" = [
	{
		"name"		= "-p, --poetry-file",
		"help"		= "poetry file",
		"required" 	= true
	}
]
"""

class Application:
	def __init__(self):
		self.args = None
		pass

	def __del__(self):
		pass

	## 初始化应用程序
	def init(self) -> None:
		self.args = config.args_parse(__ARGS_INFO__)
		config.setup_logger("poetry", "app")
		log.debug("args: %s", self.args)

	## 启动模块
	def run(self) -> None:
		log.info("poetry app run...")
		# 生成Token
		tk = tokenizer.Tokenizer()
		tk.gen_token_dict(self.args.poetry_file)
		# 定义模型
		model = tf.keras.Sequential([
			tf.keras.layers.Input((None,)),
			tf.keras.layers.Embedding(input_dim = tk.vocab_size, output_dim=128),
			tf.keras.layers.LSTM(128, dropout=0.5, return_sequences=True),
			tf.keras.layers.LSTM(128, dropout=0.5, return_sequences=True),
			tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(tk.vocab_size, activation='softmax')),
		])
		model.summary()
		# 配置优化器和损失函数
		model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy)
		# 创建数据集
		dg = PoetryDataGenerator(tk.get_poetry(), setting.BATCH_SIZE, tk, random=True)
		# 开始训练
		model.fit_generator(dg.for_fit(), steps_per_epoch=dg.steps, epochs=setting.TRAIN_EPOCHS, callbacks=[Evaluate(model, tk)])
		log.info("done!")

	## 关闭
	def kill(self):
		pass

# 
# 主进程
if __name__ == "__main__":
	try:
		app = Application()
		app.init()
		app.run()
	except KeyboardInterrupt:
		log.info("catch ctrl-c exception, abort!")
		app.kill()










