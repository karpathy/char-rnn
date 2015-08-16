--[[
A quick patch for converting GPU checkpoints to 
CPU checkpoints until I implement a more long-term
solution. Takes the path to the model and creates
a file in the same location and path, but with _cpu.t7
appended.
]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'lfs'

require 'util.OneHot'
require 'util.misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
cmd:argument('-model','GPU model checkpoint to convert')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    else
    	print('Error, no GPU available?')
        os.exit()
    end
end

-- check that clnn/cltorch are installed if user wants to use OpenCL
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        print('using OpenCL on GPU ' .. opt.gpuid .. '...')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    else
        print('Error, no GPU available?')
        os.exit()
    end
end

print('loading ' .. opt.model)
checkpoint = torch.load(opt.model)
protos = checkpoint.protos

-- convert the networks to be CPU models
for k,v in pairs(protos) do
	print('converting ' .. k .. ' to CPU')
	protos[k]:double()
end

local savefile = opt.model .. '_cpu.t7' -- append "cpu.t7" to filename
torch.save(savefile, checkpoint)
print('saved ' .. savefile)


