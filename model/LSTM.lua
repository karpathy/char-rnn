
local LSTM = {}

lstmu = require 'model.unit'

function LSTM.lstm(input_size, rnn_size, n, dropout)
  dropout = dropout or 0 

  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1, n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, size
  local outputs = {}

  for L = 1,n do
    local prev_h = inputs[L * 2 + 1]
    local prev_c = inputs[L * 2]

    if L == 1 then 
      x = OneHot(input_size)(inputs[1])
      size = input_size
    else 
      x = outputs[(L - 1) * 2] 

      if dropout > 0 then 
        x = nn.Dropout(dropout)(x) 
      end

      size = rnn_size
    end
    -- evaluate the input sums at once for efficiency

    next_c, next_h = lstmu(x, prev_c, prev_h, size, rnn_size, L)
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return LSTM

