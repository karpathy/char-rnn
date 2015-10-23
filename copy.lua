--[[

  Training a NTM to memorize input.

  The current version seems to work, giving good output after 5000 iterations
  or so. Proper initialization of the read/write weights seems to be crucial
  here.

--]]

require('../')
require('./util')
require('optim')
require('sys')

torch.manualSeed(0)

-- NTM config
local config = {
  input_dim = 10,
  output_dim = 10,
  mem_rows = 128,
  mem_cols = 20,
  cont_dim = 100
}

local input_dim = config.input_dim
local start_symbol = torch.zeros(input_dim)
start_symbol[1] = 1
local end_symbol = torch.zeros(input_dim)
end_symbol[2] = 1

function generate_sequence(len, bits)
  local seq = torch.zeros(len, bits + 2)
  for i = 1, len do
    seq[{i, {3, bits + 2}}] = torch.rand(bits):round()
  end
  return seq
end

function forward(model, seq, print_flag)
  local len = seq:size(1)
  local loss = 0

  -- present start symbol
  model:forward(start_symbol)

  -- present inputs
  if print_flag then print('write head max') end
  for j = 1, len do
    model:forward(seq[j])
    if print_flag then print_write_max(model) end
  end

  -- present end symbol
  model:forward(end_symbol)

  -- present targets
  local zeros = torch.zeros(input_dim)
  local outputs = torch.Tensor(len, input_dim)
  local criteria = {}
  if print_flag then print('read head max') end
  for j = 1, len do
    criteria[j] = nn.BCECriterion()
    outputs[j] = model:forward(zeros)
    loss = loss + criteria[j]:forward(outputs[j], seq[j]) * input_dim
    if print_flag then print_read_max(model) end
  end
  return outputs, criteria, loss
end

function backward(model, seq, outputs, criteria)
  local len = seq:size(1)
  local zeros = torch.zeros(input_dim)
  for j = len, 1, -1 do
    model:backward(
      zeros,
      criteria[j]
        :backward(outputs[j], seq[j])
        :mul(input_dim)
      )
  end

  model:backward(end_symbol, zeros)
  for j = len, 1, -1 do
    model:backward(seq[j], zeros)
  end
  model:backward(start_symbol, zeros)
end

local model = ntm.NTM(config)
local params, grads = model:getParameters()

local num_iters = 10000
local start = sys.clock()
local print_interval = 25
local min_len = 1
local max_len = 20

print(string.rep('=', 80))
print("NTM copy task")
print('training up to ' .. num_iters .. ' iteration(s)')
print('min sequence length = ' .. min_len)
print('max sequence length = ' .. max_len)
print(string.rep('=', 80))
print('num params: ' .. params:size(1))

local rmsprop_state = {
  learningRate = 1e-4,
  momentum = 0.9,
  decay = 0.95
}

-- local adagrad_state = {
--   learningRate = 1e-3
-- }

-- train
for iter = 1, num_iters do
  local print_flag = (iter % print_interval == 0)
  local feval = function(x)
    if print_flag then
      print(string.rep('-', 80))
      print('iter = ' .. iter)
      print('learn rate = ' .. rmsprop_state.learningRate)
      print('momentum = ' .. rmsprop_state.momentum)
      print('decay = ' .. rmsprop_state.decay)
      printf('t = %.1fs\n', sys.clock() - start)
    end

    local loss = 0
    grads:zero()

    local len = math.floor(torch.random(min_len, max_len))
    local seq = generate_sequence(len, input_dim - 2)
    local outputs, criteria, sample_loss = forward(model, seq, print_flag)
    loss = loss + sample_loss
    backward(model, seq, outputs, criteria)
    if print_flag then
      print("target:")
      print(seq)
      print("output:")
      print(outputs)
    end

    -- clip gradients
    grads:clamp(-10, 10)
    if print_flag then
      print('max grad = ' .. grads:max())
      print('min grad = ' .. grads:min())
      print('loss = ' .. loss)
    end
    return loss, grads
  end

  --optim.adagrad(feval, params, adagrad_state)
  ntm.rmsprop(feval, params, rmsprop_state)
end
