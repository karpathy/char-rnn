local SeqModel = { }
SeqModel.__index = SeqModel

function SeqModel.build(modelType, vocab_size, rnn_size, num_layers, dropout)
    local protos = {}

    if modelType == 'lstm' then
        local LSTM = require 'model.LSTM'
        protos.rnn = LSTM.lstm(vocab_size, rnn_size, num_layers, dropout)
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


function SeqModel.new(model, state)
  local init_state_global = clone_list(state)
  local o = {}

  setmetatable(o, SeqModel)

  o.model = model
  o.init_state = state
  o.init_state_global = init_state_global

  return o
end

function SeqModel:forward(rnn_state, x, y)
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

    return predictions
end

function SeqModel:backward(rnn_state, predictions, x, y)
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[self.model.seq_length] = clone_list(self.init_state, true)} -- true also zeros the clones

    for t = self.model.seq_length, 1, -1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = self.model.criterion[t]:backward(predictions[t], y[t])
        table.insert(drnn_state[t], doutput_t)
        local dlst = self.model.rnn[t]:backward({x[t], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the 
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-1] = v
            end
        end
    end
end

function SeqModel:loss(predictions, y)
    local loss = 0

    for t = 1, self.model.seq_length do
        loss = loss + self.model.criterion[t]:forward(predictions[t], y[t])
    end

    return loss / self.model.seq_length
end

return SeqModel
