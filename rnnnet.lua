local model_utils = require 'util.model_utils'

function initParams(rnn, do_random_init, model, num_layers, rnn_size)
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

function initState(num_layers, batch_size, rnn_size, model)
  local state = {}

  for L = 1, num_layers do
      local h_init = torch.zeros(batch_size, rnn_size)
      h_init = transferGpu(h_init)

      table.insert(state, h_init:clone())
      if model == 'lstm' then
          table.insert(state, h_init:clone())
      end
  end

  return state
end


