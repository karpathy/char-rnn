require "./Print.lua"
local LSTMEX = {}

-- retunrs memory_dim vector - convex combination of memory slots
function LSTMEX.ReadHead(name, control, control_dim, memory, memory_slots, memory_dim)
  print(string.format(
    '%d guided attention reader on %dx%d memory matrix initialized...',
    control_dim,
    memory_slots,
    memory_dim
  ))
  local transform = nn.Linear(control_dim, memory_slots)(control)
  local address = nn.SoftMax()(transform):annotate{name=name}
  address = nn.Reshape(memory_slots,1,true)(address)
  local hologram = nn.MM(true){memory, address}
  return nn.Reshape(memory_dim,true)(hologram)
  --return nn.Tanh()(hologram)
end

-- writes to whole memory weighted decomposition of x ruled by y signal
function LSTMEX.WriteHead(name, control, control_dim, x, memory, memory_slots, memory_dim)
  print(string.format(
    '%d guided writer on %dx%d memory matrix initialized...',
    control_dim,
    memory_slots,
    memory_dim
  ))
  local transform = nn.Linear(control_dim, memory_slots)(control)
  local address = nn.SoftMax()(transform):annotate{name=name}
  address = nn.Reshape(memory_slots,1,true)(address)
  --address = nn.Print('address',true)(address)

  local tx = nn.Reshape(1,memory_dim,true)(x)
  --tx = nn.Print('x')(tx)
  local delta = nn.MM(){address, tx}
  --delta = nn.Print('delta')(delta)
  local updated_memory = nn.CAddTable()({delta, memory})
  return updated_memory
end

-- Gated eraser by control signal
function LSTMEX.EraseHead(name, control, control_dim, memory, memory_slots, memory_dim)
  print(string.format(
    '%d guided eraser on %dx%d memory matrix initialized...',
    control_dim,
    memory_slots,
    memory_dim
  ))
  --control = nn.Print('erase_signal', true)(control)
  local transform = nn.Linear(control_dim, memory_slots)(control)
  --transform = nn.Print('transformed_erase')(transform)
  local address = nn.SoftMax()(transform):annotate{name=name}
  --address = nn.Print('erase_address')(address)
  address = nn.AddConstant(1,false)(nn.MulConstant(-1,false)(address))
  --address = nn.Print('mul_mask', true)(address)
  address = nn.Replicate(memory_dim,3,3)(address)
  --address = nn.Print('replicated_mask')(address)
  local updated_memory = nn.CMulTable()({address, memory})
  return updated_memory
end


-- writes to whole memory weighted decomposition of x ruled by y signal
function LSTMEX.EraseHeadModule(name, control_dim, memory_slots, memory_dim)
  local control = nn.Identity()()
  local memory = nn.Identity()()
  local updated_memory = LSTMEX.EraseHead(name,control,control_dim,memory,memory_slots,memory_dim)
  return nn.gModule({control,memory}, {updated_memory})
end
-- writes to whole memory weighted decomposition of x ruled by y signal
function LSTMEX.WriteHeadModule(name, control_dim, memory_slots, memory_dim)
  local control = nn.Identity()()
  local x = nn.Identity()()
  local memory = nn.Identity()()
  local updated_memory = LSTMEX.WriteHead(name,control,control_dim,x,memory,memory_slots,memory_dim)
  return nn.gModule({control,x,memory}, {updated_memory})
end

-- writes to whole memory weighted decomposition of x ruled by y signal
function LSTMEX.WriteEraseHead(name, i, f, x, i_dim, f_dim, x_dim, memory, memory_slots, memory_dim)
  print(string.format(
    'WriteErase head %d guided writer on %dx%d memory matrix initialized...',
    i_dim,
    memory_slots,
    memory_dim
  ))

  local write_weights = nn.SoftMax()(nn.Linear(i_dim, memory_slots)(i))
  local delta = nn.MM(){ -- memory_slots X memory_dim matrix
    nn.Reshape(memory_slots,1,true)(write_weights),
    nn.Reshape(1,memory_dim,true)(x)
  }


  --delta = nn.Print('delta')(delta)
  --local updated_memory = nn.CAddTable()({delta, memory})

  --control = nn.Print('erase_signal', true)(control)
  local T = nn.SoftMax()(nn.Linear(f_dim, memory_slots)(f))
  T = nn.Replicate(memory_dim,3,3)(T)
  local M = nn.CAddTable()({
    nn.CMulTable()({T,memory}),
    nn.CMulTable()({
      delta,
      nn.AddConstant(1,false)(nn.MulConstant(-1,false)(T))
    })
  })

  return M
end


function LSTMEX.lstm(input_size, rnn_size, n, dropout, memory_slots)
  dropout = dropout or 0

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then
      x = OneHot(input_size)(inputs[1])
    --x = nn.Print(x)
      input_size_L = input_size
    else
      x = outputs[(L-1)*2]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})
    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update

    -- erase controlled by forget gate
    local erased_c = LSTMEX.EraseHead('Erase',forget_gate,rnn_size,prev_c,memory_slots,rnn_size)

    -- write controlled by input gate
    local next_c = LSTMEX.WriteHead('Write',in_gate,rnn_size,in_transform,erased_c,memory_slots,rnn_size)
    next_c = nn.PrintTensor(10,"Memory")(next_c)
    --local next_c           = nn.CAddTable()({
    --    nn.CMulTable()({forget_gate, prev_c}),
    --    nn.CMulTable()({in_gate,     in_transform})
    --  })
    --next_c = nn.Print()(next_c)
    -- read controlled by output gate
    local next_h = LSTMEX.ReadHead('Read', out_gate, rnn_size, next_c, memory_slots, rnn_size)
    next_h = nn.Tanh()(next_h)
    --local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

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

function LSTMEX.lstm2(input_size, rnn_size, n, dropout, memory_slots)
  dropout = dropout or 0

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then
      x = OneHot(input_size)(inputs[1])
    --x = nn.Print(x)
      input_size_L = input_size
    else
      x = outputs[(L-1)*2]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})
    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c = LSTMEX.WriteEraseHead(
      'WriteErase',
      in_gate,
      forget_gate,
      in_transform,
      rnn_size,               --|
      rnn_size,               --| HUooooooooooH!
      rnn_size,               --|
      prev_c,
      memory_slots,
      rnn_size
    )
    next_c = nn.PrintTensor(10,"Memory")(next_c)
    --next_c = nn.Print("next_c")(next_c)
    -- erase controlled by forget gate
    --local erased_c = LSTMEX.EraseHead('Erase',forget_gate,rnn_size,prev_c,memory_slots,rnn_size)

    -- write controlled by input gate
    --local next_c = LSTMEX.WriteHead('Write',in_gate,rnn_size,in_transform,erased_c,memory_slots,rnn_size)
    --local next_c           = nn.CAddTable()({
    --    nn.CMulTable()({forget_gate, prev_c}),
    --    nn.CMulTable()({in_gate,     in_transform})
    --  })
    --next_c = nn.Print()(next_c)
    -- read controlled by output gate
    local next_h = LSTMEX.ReadHead('Read', out_gate, rnn_size, next_c, memory_slots, rnn_size)
    next_h = nn.Tanh()(next_h)
    --local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

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

return LSTMEX
