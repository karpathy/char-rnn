-- simple script that loads a checkpoint and prints its opts

require 'torch'
require 'nn'
require 'nngraph'

require 'util.OneHot'
require 'util.misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Load a checkpoint and print its options and validation losses.')
cmd:text()
cmd:text('Options')
cmd:argument('-model','model to load')
cmd:option('-gpuid',0,'gpu to use')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

if opt.gpuid >= 0 and opt.opencl == 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1)
end

if opt.gpuid >= 0 and opt.opencl == 1 then
    print('using OpenCL on GPU ' .. opt.gpuid .. '...')
    require 'cltorch'
    require 'clnn'
    cltorch.setDevice(opt.gpuid + 1)
end

local model = torch.load(opt.model)

print('opt:')
print(model.opt)
print('val losses:')
print(model.val_losses)

