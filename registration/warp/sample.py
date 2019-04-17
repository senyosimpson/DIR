import torch.nn.functional as F
from torch.autograd import Variable

def grid_sample(x, grid, canvas=None):
    output = F.grid_sample(x, grid)
    if canvas is None:
        return output
    else:
        x_mask = Variable(x.data.new(x.size()).fill_(1))
        output_mask = F.grid_sample(x_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output
