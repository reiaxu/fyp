import torch
from torch import nn
import math


class Quantizers(nn.Module):
	def __init__(self, bw, quant_mode = 'minmax', act_q = True, quantize = False):
		super(Quantizers, self).__init__()
		self.is_quantize = quantize
		self.act_q = act_q
		self.init = False
		self.is_symmetric = True
		self.quant_mode = quant_mode
		self.calibration = False
		self.n = bw
		self.offset = None
		self.min = torch.Tensor([float('inf')])[0].cuda()
		self.max = torch.Tensor([float('-inf')])[0].cuda()
		self.scale = None

	def set_quantize(self, flag):
		self.is_quantize = flag

	def estimate_range(self, flag):
		self.calibration = flag

	def init_params(self, x_f):
		'''
		https://heartbeat.fritz.ai/quantization-arithmetic-421e66afd842
		
		There exist two modes
		1) Symmetric:
			Symmetric quantization uses absolute max value as its min/max meaning symmetric with respect to zero
		2) Asymmetric
			Asymmetric Quantization uses actual min/max, meaning it is asymmetric with respect to zero 
		
		Scale factor uses full range [-2**n / 2, 2**n - 1]
		'''
		if self.is_symmetric:
			x_min, x_max = -torch.max(torch.abs(x_f)), torch.max(torch.abs(x_f))
		else:
			x_min, x_max = torch.min(x_f), torch.max(x_f)
		
		self.min = torch.min(x_min, self.min)
		self.max = torch.max(x_max, self.max)
		max_range = self.max - self.min
		
		self.scale = max_range / float(2**self.n - 1)
		if not self.is_symmetric:
				self.offset = torch.round(-x_min / self.scale)
		self.init = True

	def quant_dequant(self, x_f, scale , offset):
		'''
		Quantizing
		Formula is derived from below:
		https://medium.com/ai-innovation/quantization-on-pytorch-59dea10851e1
		'''
		if self.n == 1:
			x_int = torch.sign( x_f  / scale )
		else:
			x_int = torch.round( x_f  / scale )

		if not self.is_symmetric:
			x_int += offset

		if self.is_symmetric:
			l_bound, u_bound = -2**(self.n - 1), 2**(self.n-1) - 1
		else:   
			l_bound, u_bound = 0, 2**(self.n) - 1
		if self.n == 1:
			x_q = x_int
		else:
			x_q = torch.clamp(x_int, min = l_bound, max = u_bound)
		'''
		De-quantizing
		'''
		if not self.is_symmetric:
			x_q -= offset
		x_float_q = x_q * scale
		return x_float_q

	def forward(self, x_f):
		if (self.calibration and self.act_q) or not self.init:
			self.init_params(x_f)
		return self.quant_dequant(x_f, self.scale, self.offset) if self.is_quantize else x_f


class QuantConv(nn.Module):
	def __init__(self, args, conv):
		super(QuantConv, self).__init__()
		self.conv = conv
		self.weight_quantizer = Quantizers(args.bitwidth, args.quant_scheme, act_q = False)
		self.kwarg = {	'stride' : self.conv.stride, \
						'padding' : self.conv.padding, \
						'dilation' : self.conv.dilation, \
						'groups': self.conv.groups}
		self.activation_function = None
		self.act_quantizer = Quantizers(args.bitwidth, args.quant_scheme)
		self.pre_activation = False

	def batchnorm_folding(self):
		'''
		https://towardsdatascience.com/speed-up-inference-with-batch-normalization-folding-8a45a83a89d8

		W_fold = gamma * W / sqrt(var + eps)
		b_fold = (gamma * ( bias - mu ) / sqrt(var + eps)) + beta
		'''
		if hasattr(self.conv, 'gamma'):
			gamma = getattr(self.conv, 'gamma')
			beta = getattr(self.conv, 'beta')
			mu = getattr(self.conv, 'mu')
			var = getattr(self.conv, 'var')
			eps = getattr(self.conv, 'eps')

			denom = gamma.div(torch.sqrt(var + eps))

			if getattr(self.conv, 'bias') == None:
				self.conv.bias = torch.nn.Parameter(var.new_zeros(var.shape))
			b_fold = denom*(self.conv.bias.data - mu) + beta
			self.conv.bias.data.copy_(b_fold)

			denom = denom.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
			self.conv.weight.data.mul_(denom)

	def get_params(self):
		w = self.conv.weight.detach()
		if self.conv.bias != None:
			b = self.conv.bias.detach()
		else:
			b = None
		w = self.weight_quantizer(w)
		return w, b

	def turn_preactivation_on(self):
		self.pre_activation = True

	def forward(self, x):
		w, b = self.get_params()
		out = nn.functional.conv2d(input = x, weight = w, bias = b, **self.kwarg)
		if self.activation_function and not self.pre_activation:
			out = self.activation_function(out)
		out = self.act_quantizer(out)
		return out

class QuantLinear(nn.Module):
	def __init__(self, args, linear):
		super(QuantLinear, self).__init__()
		self.fc = linear
		self.weight_quantizer = Quantizers(args.bitwidth, args.quant_scheme, act_q = False)
		self.activation_function = None
		self.act_quantizer = Quantizers(args.bitwidth, args.quant_scheme)

	def get_params(self):
		w = self.fc.weight.detach()
		if self.fc.bias != None:
			b = self.fc.bias.detach()
		else:
			b = None
		w = self.weight_quantizer(w)
		return w, b

	def forward(self, x):
		w, b = self.get_params()
		out = nn.functional.linear(x, w, b)
		if self.activation_function:
			out = self.activation_function(out)
		out = self.act_quantizer(out)
		return out

class PassThroughOp(nn.Module):
	def __init__(self):
		super(PassThroughOp, self).__init__()

	def forward(self, x):
		return x
