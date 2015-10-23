--[[

  Input: a table of 2 or 3 vectors.

  Output: the outer product of the vectors.

--]]

local OuterProd, parent = torch.class('nn.OuterProd', 'nn.Module')

function OuterProd:__init()
  parent.__init(self)
  self.gradInput = {}
end

function OuterProd:updateOutput(input)
  local order = #input
  self.order = order
  if order == 2 then
    self.output:set(torch.ger(input[1], input[2]))
    self.size = self.output:size()
  elseif order == 3 then
    -- allocate
    self.size = torch.LongStorage(order)
    local idx = 1
    for i = 1, order do
      self.size[i] = input[i]:size(1)
    end
    self.output:resize(self.size):zero()

    local u, v, w = unpack(input)
    local uv = torch.ger(u, v)
    for i = 1, self.size[3] do
      self.output[{{}, {}, i}]:add(w[i], uv)
    end
  else
    error('outer products of order higher than 3 unsupported')
  end
  return self.output
end

function OuterProd:updateGradInput(input, gradOutput)
  local order = #input
  for i = 1, order do
    self.gradInput[i] = self.gradInput[i] or input[1].new()
    self.gradInput[i]:resizeAs(input[i])
  end

  if order == 2 then
    self.gradInput[1]:copy(gradOutput * input[2])
    self.gradInput[2]:copy(gradOutput:t() * input[1])
  else
    local u, v, w = unpack(input)
    local du, dv, dw = u:size(1), v:size(1), w:size(1)
    local uv = input[1].new():resize(du, dv):zero()
    for i = 1, dw do
      uv:add(w[i], gradOutput[{{}, {}, i}])
    end
    self.gradInput[1]:copy(uv * input[2])
    self.gradInput[2]:copy(uv:t() * input[1])

    local vw = input[1].new():resize(dv, dw):zero()
    for i = 1, du do
      vw:add(u[i], gradOutput[{i, {}, {}}])
    end
    self.gradInput[3]:copy(vw:t() * input[2])
  end
  return self.gradInput
end
