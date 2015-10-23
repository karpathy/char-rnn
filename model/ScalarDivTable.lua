--[[

Input: A table {x, y} of a Tensor and a scalar.

Output: x / y

--]]


local ScalarDivTable, parent = torch.class('nn.ScalarDivTable', 'nn.Module')

function ScalarDivTable:__init()
  parent.__init(self)
  self.gradInput = {}
end

function ScalarDivTable:updateOutput(input)
  local v, scale = unpack(input)
  return self.output:set(v / scale[1])
end

function ScalarDivTable:updateGradInput(input, gradOutput)
  local v, scale = unpack(input)
  self.gradInput[1] = self.gradInput[1] or input[1].new()
  self.gradInput[2] = self.gradInput[2] or input[2].new()
  self.gradInput[2]:resizeAs(input[2])

  local c = scale[1]
  self.gradInput[1]:set(gradOutput / c)
  self.gradInput[2][1] = -gradOutput:dot(v) / (c * c)
  return self.gradInput
end
