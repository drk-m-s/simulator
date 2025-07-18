import torch

import upmem_llm_framework.pytorch_upmem_layers as upmem_layers
upmem_layers.profiler_init()

class TinyModel(torch.nn.Module):

	def __init__(self):
		super(TinyModel, self).__init__()
		
		self.linear1 = torch.nn.Linear(100, 200)
		self.activation = torch.nn.ReLU()
		self.linear2 = torch.nn.Linear(200, 10)
		#self.softmax = torch.nn.Softmax(dim = 0)
	
	def forward(self, x):
		x = self.linear1(x)
		x = self.activation(x)
		x = self.linear2(x)
		x = torch.nn.functional.softmax(x, dim = 0)
		return x

tinymodel = TinyModel()

print('The model:')
print(tinymodel)

myTensor = torch.rand(100)

layer_mapping = {
        "linear1":"PIM-AI-1chip",
        "linear2":"PIM-AI-1chip",
        }

upmem_layers.profiler_start(layer_mapping, last_layer = "linear2")
prediction = tinymodel.forward(myTensor)
upmem_layers.profiler_end()
print (prediction)
