import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.onnx

import onnx
from onnx_tf.backend import prepare


def load_torch_model(weight_dir, weight):
	model_name = weight_dir + str(weight) + '_best_model.pt'
	model = torch.load(model_name)
	model.eval()
	return model

def save_model_as_onxx(model, dummy_input, weight_dir, weight):
	model_name = weight_dir + str(weight) + '_best_model.onnx'
	print("exporting...")
	torch.onnx.export(model, dummy_input, model_name)

	return model_name

	# print("done exporting")
	# print("loading into onnx")
	# onnx_model = onnx.load(model_name)

	# print("preparing...")
	# tf_rep = prepare(onnx_model)
	# print('exporting...')
	# tf_rep.export_graph(weight_dir + str(weight) + "_best_model.pb")


