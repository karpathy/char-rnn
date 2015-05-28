-- simple script that loads a checkpoint and prints its opts

require 'torch'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'

require 'util.OneHot'
require 'util.misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Load a checkpoint and print its options and validation losses.')
cmd:text()
cmd:text('Options')
cmd:argument('-model','model to load')
cmd:option('-gpuid',0,'gpu to use')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

print('using CUDA on GPU ' .. opt.gpuid .. '...')
require 'cutorch'
require 'cunn'
cutorch.setDevice(opt.gpuid + 1)

local model = torch.load(opt.model)

print('opt:')
print(model.opt)
print('val losses:')
print(model.val_losses)

