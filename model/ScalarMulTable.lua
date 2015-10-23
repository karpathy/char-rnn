--[[

 Input: A table {x, y} of a Tensor x and a scalar y.

 Output: x * y

--]]

local ScalarMulTable, parent = torch.class('nn.ScalarMulTable', 'nn.Module')

function ScalarMulTable:__init()
  parent.__init(self)
  self.gradInput = {}
end

function ScalarMulTable:updateOutput(input)
  local v, scale = unpack(input)
  return self.output:set(v * scale[1])
end

function ScalarMulTable:updateGradInput(input, gradOutput)
  local v, scale = unpack(input)
  self.gradInput[1] = self.gradInput[1] or input[1].new()
  self.gradInput[2] = self.gradInput[2] or input[2].new()
  self.gradInput[2]:resizeAs(input[2])

  self.gradInput[1]:set(gradOutput * scale[1])
  self.gradInput[2][1] = gradOutput:dot(v)
  return self.gradInput
end
