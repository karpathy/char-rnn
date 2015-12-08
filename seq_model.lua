local model_utils = require 'util.model_utils'

local SeqModel = { }
SeqModel.__index = SeqModel

function SeqModel.buildProto(modelType, vocab_size, rnn_size, num_layers, dropout)
    local protos = {}

    if modelType == 'lstm' then
        local LSTM = require 'model.LSTM'
        protos.rnn = LSTM.lstm(vocab_size, rnn_size, num_layers, dropout, OneHot(vocab_size))
    elseif modelType == 'gru' then
        local GRU = require 'model.GRU'
        protos.rnn = GRU.gru(vocab_size, rnn_size, num_layers, dropout)
    elseif modelType == 'rnn' then
        local RNN = require 'model.RNN'
        protos.rnn = RNN.rnn(vocab_size, rnn_size, num_layers, dropout)
    end

    protos.criterion = nn.ClassNLLCriterion()

    return protos
end

-- make a bunch of clones after flattening, as that reallocates memory
local function buildSeq(protos, seq_length)
  local model = {}

  for name, proto in pairs(protos) do
      print('cloning ' .. name)
      model[name] = model_utils.clone_many_times(proto, seq_length)
  end

  model.seq_length = seq_length

  return model
end

-- local
function initState(num_layers, batch_size, rnn_size, modelType)
  local state = {}

  for L = 1, num_layers do
      local h_init = transferGpu(torch.zeros(batch_size, rnn_size))

      table.insert(state, h_init:clone())
      if modelType == 'lstm' then
          table.insert(state, h_init:clone())
      end
  end

  return state
end

local function initParams(rnn, do_random_init, model, num_layers, rnn_size)
  local params, grad_params = model_utils.combine_all_parameters(rnn)

  -- initialization
  if do_random_init then
      params:uniform(-0.08, 0.08) -- small uniform numbers
  end

  -- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
  if model == 'lstm' then
      for layer_idx = 1, num_layers do
          for _,node in ipairs(protos.rnn.forwardnodes) do
              if node.data.annotations.name == "i2h_" .. layer_idx then
                  print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
                  -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
                  node.data.module.bias[{{rnn_size+1, 2*rnn_size}}]:fill(1.0)
              end
          end
      end
  end

  return params, grad_params
end



function SeqModel.new(protos, seq_length, num_layers, batch_size, rnn_size, modelType, vocab)
  -- init params
  local params, grad_params = initParams(protos.rnn, do_random_init, modelType, num_layers, rnn_size)

  -- init state
  local state = initState(num_layers, batch_size, rnn_size, modelType)
  local init_state_global = clone_list(state)

  -- init model
  local model = buildSeq(protos, seq_length)

  -- init class
  local o = {}

  setmetatable(o, SeqModel)
  o.model = model
  o.init_state = state
  o.init_state_global = init_state_global
  o.params = params
  o.grad_params = grad_params

  o.protos = protos
  o.rnn_size = rnn_size
  o.num_layers = num_layers
  o.model_type = modelType
  o.vocab = vocab
  o.batch_size = batch_size

  return o
end

function SeqModel:forward(x)
    local rnn_state = {[0] = self.init_state_global}
    local predictions = {} -- softmax outputs

    for t = 1, self.model.seq_length do
        self.model.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)

        local lst = self.model.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}

        rnn_state[t] = {}
        for i = 1, #self.init_state do 
          table.insert(rnn_state[t], lst[i]) 
        end -- extract the state, without output

        predictions[t] = lst[#lst] -- last element is the prediction
    end

    self.rnn_state = rnn_state

    return predictions, rnn_state
end

function SeqModel:backward(predictions, x, y)
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[self.model.seq_length] = clone_list(self.init_state, true)} -- true also zeros the clones

    for t = self.model.seq_length, 1, -1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = self.model.criterion[t]:backward(predictions[t], y[t])
        table.insert(drnn_state[t], doutput_t)
        local dlst = self.model.rnn[t]:backward({x[t], unpack(self.rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the 
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-1] = v
            end
        end
    end

    -- transfer final state to initial state (BPTT)
    self.init_state_global = self.rnn_state[#self.rnn_state] -- NOTE: I don't think this needs to be a clone, right?
end

function SeqModel:loss(predictions, y)
    local loss = 0

    for t = 1, self.model.seq_length do
        loss = loss + self.model.criterion[t]:forward(predictions[t], y[t])
    end

    return loss / self.model.seq_length
end

-- evaluate the loss over an entire split
function SeqModel:eval(ds, split_index)
    print('evaluating loss over split index ' .. split_index)
    local n = ds.split_sizes[split_index]

    ds:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0

    -- iterate over batches in the split
    for i = 1, n do         
        local x, y = prepro(ds:next_batch(split_index))
        local pred = self:forward(x, y)
        loss = loss + self:loss(pred, y) 

        print(i .. "/" .. n)
    end

    return loss / n
end

function SeqModel:save(fileName)
  local checkpoint = {}
  checkpoint.protos = self.protos
  checkpoint.rnn_size = self.rnn_size
  checkpoint.num_layers = self.num_layers
  checkpoint.model_type = self.model_type
  checkpoint.vocab = self.vocab

  torch.save(fileName, checkpoint)
end

return SeqModel
