import torch
import math


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):

    @staticmethod
    def forward(input_tensor, consensus_type, dim=1):
        shape = input_tensor.size()
        if consensus_type == 'avg':
            output = input_tensor.mean(dim=dim, keepdim=True)
        elif consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        #return output, shape
        return output

    @staticmethod
    def backward(grad_output, consensus_type, shape, dim):
        if consensus_type == 'avg':
            grad_in = grad_output.expand(shape) / float(shape[dim])
        elif consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus.forward(input, self.consensus_type, self.dim)
