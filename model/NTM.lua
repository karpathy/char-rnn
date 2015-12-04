--[[

  Implementation of the Neural Turing Machine described here:

  http://arxiv.org/pdf/1410.5401v2.pdf

  Variable names take after the notation in the paper. Identifiers with "r"
  appended indicate read-head variables, and likewise for those with "w" appended.

  The NTM take a configuration table at initialization time with the following
  options:

  * input_dim   dimension of input vectors (required)
  * output_dim  dimension of output vectors (required)
  * mem_rows    number of rows of memory
  * mem_cols    number of columns of memory
  * cont_dim    dimension of controller state
  * cont_layers number of controller layers
  * shift_range allowed range for shifting read/write weights
  * write_heads number of write heads
  * read_heads  number of read heads

--]]
require "./SmoothCosineSimilarity"
require "./ScalarMulTable"
require "./CircularConvolution"
require "./PowTable"
require "./NormalizeBySum"
require "./OuterProd"
require "./Squeeze"

local function share_params(cell, src, ...)
  for i = 1, #cell.forwardnodes do
    local node = cell.forwardnodes[i]
    if node.data.module then
      node.data.module:share(src.forwardnodes[i].data.module, ...)
    end
  end
end

local NTM, parent = torch.class('nn.NTM', 'nn.Module')

function NTM:__init(config)
  self.input_dim   = config.input_dim   or error('config.input_dim must be specified')
  self.output_dim  = config.output_dim  or error('config.output_dim must be specified')
  self.mem_rows    = config.mem_rows    or 20
  self.mem_cols    = config.mem_cols    or 128
  self.cont_dim    = config.cont_dim    or 128
  self.cont_layers = config.cont_layers or 1
  self.shift_range = config.shift_range or 1
  self.write_heads = config.write_heads or 1
  self.read_heads  = config.read_heads  or 1

  self.depth = 0
  self.cells = {}
  self.master_cell = self:new_cell()
  self.init_module = self:new_init_module()

  self:init_grad_inputs()
end

function NTM:init_grad_inputs()
  local ww_gradInput
  if self.write_heads == 1 then
    ww_gradInput = torch.zeros(self.mem_rows)
  else
    ww_gradInput = {}
    for i = 1, self.write_heads do
      ww_gradInput[i] = torch.zeros(self.mem_rows)
    end
  end

  local wr_gradInput, r_gradInput
  if self.read_heads == 1 then
    wr_gradInput = torch.zeros(self.mem_rows)
    r_gradInput = torch.zeros(self.mem_cols)
  else
    wr_gradInput, r_gradInput = {}, {}
    for i = 1, self.read_heads do
      wr_gradInput[i] = torch.zeros(self.mem_rows)
      r_gradInput[i] = torch.zeros(self.mem_cols)
    end
  end

  local m_gradInput, c_gradInput
  if self.cont_layers == 1 then
    m_gradInput = torch.zeros(self.cont_dim)
    c_gradInput = torch.zeros(self.cont_dim)
  else
    m_gradInput, c_gradInput = {}, {}
    for i = 1, self.cont_layers do
      m_gradInput[i] = torch.zeros(self.cont_dim)
      c_gradInput[i] = torch.zeros(self.cont_dim)
    end
  end

  self.gradInput = {
    torch.zeros(self.input_dim), -- input
    torch.zeros(self.mem_rows, self.mem_cols), -- M
    wr_gradInput,
    ww_gradInput,
    r_gradInput,
    m_gradInput,
    c_gradInput
  }
end

-- The initialization module initializes the state of NTM memory,
-- read/write weights, and the state of the LSTM controller.
function NTM:new_init_module()
  local dummy = nn.Identity()() -- always zero
  local output_init = nn.Tanh()(nn.Linear(1, self.input_dim)(dummy))

  -- memory
  local M_init_lin = nn.Linear(1, self.mem_rows * self.mem_cols)
  local M_init = nn.View(self.mem_rows, self.mem_cols)(
    nn.Tanh()(M_init_lin(dummy)))

  -- read weights
  local wr_init, r_init = {}, {}
  for i = 1, self.read_heads do
    local wr_init_lin = nn.Linear(1, self.mem_rows)
    wr_init[i] = nn.SoftMax()(wr_init_lin(dummy))
    r_init[i] = nn.Tanh()(nn.Linear(1, self.mem_cols)(dummy))

    -- We initialize the read and write distributions such that the
    -- weights decay exponentially over the rows of NTM memory.
    -- This sort of initialization seems to be important in my experiments (kst).
    wr_init_lin.bias:copy(torch.range(self.mem_rows, 1, -1))
  end

  -- write weights
  local ww_init = {}
  for i = 1, self.write_heads do
    local ww_init_lin = nn.Linear(1, self.mem_rows)
    ww_init[i] = nn.SoftMax()(ww_init_lin(dummy))

    -- See initialization comment above
    ww_init_lin.bias:copy(torch.range(self.mem_rows, 1, -1))
  end

  -- controller state
  local m_init, c_init = {}, {}
  for i = 1, self.cont_layers do
    m_init[i] = nn.Tanh()(nn.Linear(1, self.cont_dim)(dummy))
    c_init[i] = nn.Tanh()(nn.Linear(1, self.cont_dim)(dummy))
  end

  -- wrap tables as nngraph nodes
  ww_init = nn.Identity()(ww_init)
  wr_init = nn.Identity()(wr_init)
  r_init = nn.Identity()(r_init)
  m_init = nn.Identity()(m_init)
  c_init = nn.Identity()(c_init)

  local inits = {
    output_init, M_init, wr_init, ww_init, r_init, m_init, c_init
  }
  return nn.gModule({dummy}, inits)
end

-- Create a new NTM cell. Each cell shares the parameters of the "master" cell
-- and stores the outputs of each iteration of forward propagation.
function NTM:new_cell()
  -- input to the network
  local input = nn.LookupTable(self.input_dim,self.input_dim)()
  local inn = nn.Squeeze()(input)

  -- previous memory state and read/write weights
  local M_p = nn.Identity()()
  local wr_p = nn.Identity()()
  local ww_p = nn.Identity()()

  -- vector read from memory
  local r_p = nn.Identity()()

  -- LSTM controller output
  local mtable_p = nn.Identity()()
  local ctable_p = nn.Identity()()

  -- output and hidden states of the controller module
  local mtable, ctable = self:new_controller_module(inn, r_p, mtable_p, ctable_p)
  local m = (self.cont_layers == 1) and mtable
    or nn.SelectTable(self.cont_layers)(mtable)
  local M, wr, ww, r = self:new_mem_module(M_p, wr_p, ww_p, m)
  local output = self:new_output_module(m)

  local inputs = {input, M_p, wr_p, ww_p, r_p, mtable_p, ctable_p}
  local outputs = {output, M, wr, ww, r, mtable, ctable}

  local cell = nn.gModule(inputs, outputs)
  if self.master_cell ~= nil then
    share_params(cell, self.master_cell, 'weight', 'bias', 'gradWeight', 'gradBias')
  end
  return cell
end

-- Create a new LSTM controller
function NTM:new_controller_module(input, r_p, mtable_p, ctable_p)
  -- multilayer LSTM
  local mtable, ctable = {}, {}
  for layer = 1, self.cont_layers do
    local new_gate, m_p, c_p
    if self.cont_layers == 1 then
      m_p = mtable_p
      c_p = ctable_p
    else
      m_p = nn.SelectTable(layer)(mtable_p)
      c_p = nn.SelectTable(layer)(ctable_p)
    end

    if layer == 1 then
      new_gate = function()
        local in_modules = {
          nn.Linear(self.input_dim, self.cont_dim)(input),
          nn.Linear(self.cont_dim, self.cont_dim)(m_p)
        }
        if self.read_heads == 1 then
          table.insert(in_modules, nn.Linear(self.mem_cols, self.cont_dim)(r_p))
        else
          for i = 1, self.read_heads do
            local vec = nn.SelectTable(i)(r_p)
            table.insert(in_modules, nn.Linear(self.mem_cols, self.cont_dim)(vec))
          end
        end
        return nn.CAddTable()(in_modules)
      end
    else
      new_gate = function()
        return nn.CAddTable(){
          nn.Linear(self.cont_dim, self.cont_dim)(mtable[layer - 1]),
          nn.Linear(self.cont_dim, self.cont_dim)(m_p)
        }
      end
    end

    -- input, forget, and output gates
    local i = nn.Sigmoid()(new_gate())
    local f = nn.Sigmoid()(new_gate())
    local o = nn.Sigmoid()(new_gate())
    local update = nn.Tanh()(new_gate())

    -- update the state of the LSTM cell
    ctable[layer] = nn.CAddTable(){
      nn.CMulTable(){f, c_p},
      nn.CMulTable(){i, update}
    }

    mtable[layer] = nn.CMulTable(){o, nn.Tanh()(ctable[layer])}
  end

  mtable = nn.Identity()(mtable)
  ctable = nn.Identity()(ctable)
  return mtable, ctable
end

-- Create a new module to read/write to memory
function NTM:new_mem_module(M_p, wr_p, ww_p, m)
  -- read heads
  local wr, r
  if self.read_heads == 1 then
    wr, r = self:new_read_head(M_p, wr_p, m)
  else
    wr, r = {}, {}
    for i = 1, self.read_heads do
      local prev_weights = nn.SelectTable(i)(wr_p)
      wr[i], r[i] = self:new_read_head(M_p, prev_weights, m)
    end
    wr = nn.Identity()(wr)
    r = nn.Identity()(r)
  end

  -- write heads
  local ww, a, e, M_erase, M_write
  if self.write_heads == 1 then
    ww, a, e = self:new_write_head(M_p, ww_p, m)
    M_erase = nn.AddConstant(1)(nn.MulConstant(-1)(nn.OuterProd(){ww, e}))
    M_write = nn.OuterProd(){ww, a}
  else
    ww, a, e, M_erase, M_write = {}, {}, {}, {}, {}
    for i = 1, self.write_heads do
      local prev_weights = nn.SelectTable(i)(ww_p)
      ww[i], a[i], e[i] = self:new_write_head(M_p, prev_weights, m)
      M_erase[i] = nn.AddConstant(1)(nn.MulConstant(-1)(nn.OuterProd(){ww[i], e[i]}))
      M_write[i] = nn.OuterProd(){ww[i], a[i]}
    end
    M_erase = nn.CMulTable()(M_erase)
    M_write = nn.CAddTable()(M_write)
    ww = nn.Identity()(ww)
  end

  -- erase some history from memory
  --M_erase = nn.PrintTensor(1,"EraseMemory")(M_erase)
  local Mtilde = nn.CMulTable(){M_p, M_erase}
  --M_write = nn.PrintTensor(1,"WriteMemory")(M_write)
  -- write to memory
  local M = nn.CAddTable(){Mtilde, M_write}

  M = nn.PrintTensor(50,"Memory")(M)
  return M, wr, ww, r
end

function NTM:new_read_head(M_p, w_p, m)
  return self:new_head(M_p, w_p, m, true)
end

function NTM:new_write_head(M_p, w_p, m)
  return self:new_head(M_p, w_p, m, false)
end

-- Create a new head
function NTM:new_head(M_p, w_p, m, is_read)
  -- key vector
  local k     = nn.Tanh()(nn.Linear(self.cont_dim, self.mem_cols)(m))
  -- circular convolution kernel
  local s     = nn.SoftMax()(nn.Linear(self.cont_dim, 2 * self.shift_range + 1)(m))
  -- weight sharpening parameter
  local beta  = nn.SoftPlus()(nn.Linear(self.cont_dim, 1)(m))
  -- gating parameter
  local g     = nn.Sigmoid()(nn.Linear(self.cont_dim, 1)(m))
  -- exponential focusing parameter
  local gamma = nn.AddConstant(1)(
    nn.SoftPlus()(nn.Linear(self.cont_dim, 1)(m)))

  local sim = nn.SmoothCosineSimilarity(){M_p, k}
  local wc = nn.SoftMax()(nn.ScalarMulTable(){sim, beta})
  local wg = nn.CAddTable(){
    nn.ScalarMulTable(){wc, g},
    nn.ScalarMulTable(){w_p, nn.AddConstant(1)(nn.MulConstant(-1)(g))}
  }

  local wtilde = nn.CircularConvolution(){wg, s}
  local wpow = nn.PowTable(){wtilde, gamma}
  local w = nn.Normalize(2)(wpow)

  if is_read then
    local r = nn.MixtureTable(){w, M_p}
    return w, r
  else
    local a = nn.Tanh()(nn.Linear(self.cont_dim, self.mem_cols)(m))
    local e = nn.Sigmoid()(nn.Linear(self.cont_dim, self.mem_cols)(m))
    return w, a, e
  end
end

-- Create an output module, e.g. to output binary strings.
function NTM:new_output_module(m)
  local output = nn.LogSoftMax()(nn.Linear(self.cont_dim, self.output_dim)(m))
  return output
end

function NTM:r()
  self.ddepth = 0
  self.dcell = nil
  self.dprev_outputs = nil
end
-- Forward propagate one time step. The outputs of previous time steps are
-- cached for backpropagation.
function NTM:f(input)
  self.ddepth = self.ddepth or 0
  self.ddepth = self.ddepth + 1
  self.dcell = self.dcell or self:new_cell()

  local prev_outputs
  if self.ddepth == 1 then
    self.dprev_outputs = self.init_module:forward(torch.Tensor{0})
  else
    self.dprev_outputs = self.dcell.output
  end

  -- get inputs
  local inputs = {input}
  for i = 2, #self.dprev_outputs do
    inputs[i] = self.dprev_outputs[i]
  end
  --print('F',inputs)
  local outputs = self.dcell:forward(inputs)
  return outputs[1]
end

function NTM:set_last_state()
  self.prev_output = self.cells[self.depth].output
end
-- Forward propagate one time step. The outputs of previous time steps are
-- cached for backpropagation.
function NTM:forward(input)
  self.depth = self.depth + 1
  local cell = self.cells[self.depth]
  --print('forward depth', self.depth)
  if cell == nil then
    cell = self:new_cell()
    self.cells[self.depth] = cell
  end

  local prev_outputs
  if self.depth == 1 then
    --if self.prev_output == nil then
      prev_outputs = self.init_module:forward(torch.Tensor{0})
    --  self.prev_output = prev_outputs
    --else
      --print('use previous state')
      --print(self.prev_output[5][{{1,10}}])
      --prev_outputs = self.prev_output
    --end
  else
    prev_outputs = self.cells[self.depth - 1].output
  end

  -- get inputs
  local inputs = {input}
  for i = 2, #prev_outputs do
    inputs[i] = prev_outputs[i]
  end
  --print('FORWARD',inputs)
  local outputs = cell:forward(inputs)
  self.output = outputs[1]
  return self.output
end

-- Backward propagate one time step. Throws an error if called more times than
-- forward has been called.
function NTM:backward(input, grad_output)
  if self.depth == 0 then
    error("No cells to backpropagate through")
  end
  local cell = self.cells[self.depth]
  local grad_outputs = {grad_output}
  for i = 2, #self.gradInput do
    grad_outputs[i] = self.gradInput[i]
  end

  -- get inputs
  local prev_outputs
  if self.depth == 1 then
    --print("remember previous state")
    --print(self.prev_output[5][{{1,10}}])
    --prev_outputs = self.prev_output
    prev_outputs = self.init_module:forward(torch.Tensor{0})
  else
    prev_outputs = self.cells[self.depth - 1].output
  end

  local inputs = {input}
  for i = 2, #prev_outputs do
    inputs[i] = prev_outputs[i]
  end

  self.gradInput = cell:backward(inputs, grad_outputs)
  self.depth = self.depth - 1

  if self.depth == 0 then

    self.gradInput[1] = torch.zeros(self.input_dim) -- fix TODO: hm hm hm

    self.init_module:backward(torch.Tensor{0}, self.gradInput)
    for i = 1, #self.gradInput do
      local gradInput = self.gradInput[i]
      if type(gradInput) == 'table' then
        for _, t in pairs(gradInput) do
          --print(t)
          t:zero()
        end
      else
        self.gradInput[i]:zero()
      end
    end
  end
  return self.gradInput
end

-- Get the state of memory
function NTM:get_memory(depth)
  if self.depth == 0 then
    return self.initial_values[2]
  end
  local depth = depth or self.depth
  return self.cells[self.depth].output[2]
end

-- Get read head weights over the rows of memory
function NTM:get_read_weights(depth)
  if self.depth == 0 then
    return self.initial_values[3]
  end
  local depth = depth or self.depth
  return self.cells[depth].output[3]
end

-- Get write head weights over the rows of memory
function NTM:get_write_weights(depth)
  if self.depth == 0 then
    return self.initial_values[4]
  end
  local depth = depth or self.depth
  return self.cells[depth].output[4]
end

-- Get the vector read from memory
function NTM:get_read_vector(depth)
  if self.depth == 0 then
    return self.initial_values[5]
  end
  local depth = depth or self.depth
  return self.cells[depth].output[5]
end

function NTM:parameters()
  local p, g = self.master_cell:parameters()
  local pi, gi = self.init_module:parameters()
  tablex.insertvalues(p, pi)
  tablex.insertvalues(g, gi)
  return p, g
end

function NTM:forget()
  self.depth = 0
  self:zeroGradParameters()
  for i = 1, #self.gradInput do
    --print(self.gradInput[i])
    self.gradInput[i]:zero()
  end
end

function NTM:zeroGradParameters()
  self.master_cell:zeroGradParameters()
  self.init_module:zeroGradParameters()
end
