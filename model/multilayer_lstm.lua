local lstm = require 'model.lstm'

function multilayer_lstm(input_size, rnn_size, num_layers, dropout, encoder)
  dropout = dropout or 0 

  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1, num_layers do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, layer_input_size
  local outputs = {}

  for L = 1, num_layers do
    local prev_h = inputs[L * 2 + 1]
    local prev_c = inputs[L * 2]

    if L == 1 then 
      x = encoder(inputs[1])
      layer_input_size = input_size
    else 
      x = outputs[(L - 1) * 2] 

      -- activate dropout
      if dropout > 0 then 
        x = nn.Dropout(dropout)(x) 
      end

      layer_input_size = rnn_size
    end

    -- calculate outputs
    next_c, next_h = lstm(x, prev_c, prev_h, layer_input_size, rnn_size, L)
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local h2y = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(h2y)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return multilayer_lstm
