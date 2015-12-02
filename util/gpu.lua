local l_gpuid = -1
local l_opencl = false

function initGpu(gpuid, opencl, seed)
  -- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
  if gpuid >= 0 and opencl == 0 then
    initCuda(gpuid, seed)
  end

  -- initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
  if gpuid >= 0 and opencl == 1 then
    initOpenCl(gpuid, seed)
  end
end

function initCuda(gpuid, seed)
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. gpuid .. '...')
        l_gpuid = gpuid
        l_opencl = false
        cutorch.setDevice(gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        gpuid = -1 -- overwrite user setting
    end

end

function initOpenCl(gpuid, seed)
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        print('using OpenCL on GPU ' .. gpuid .. '...')
        l_gpuid = gpuid
        l_opencl = true
 
        cltorch.setDevice(gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(seed)
    else
        print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
        print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
        print('Falling back on CPU mode')
        gpuid = -1 -- overwrite user setting
    end
end

function transferGpu(t)
  if l_gpuid >= 0 and l_opencl == 0 then -- ship the input arrays to GPU
      -- have to convert to float because integers can't be cuda()'d
      return t:float():cuda()
  end

  if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
      return t:cl()
  end

  return t
end


