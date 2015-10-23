--[[

 Input: A table {x, y} of a Tensor x and a scalar y.

 Output: x^y (elementwise)

--]]

local PowTable, parent = torch.class('nn.PowTable', 'nn.Module')

function PowTable:__init()
  parent.__init(self)
  self.gradInput = {}
end

function PowTable:updateOutput(input)
  local v, p = unpack(input)
  return self.output:set(torch.pow(v, p[1]))
end

function PowTable:updateGradInput(input, gradOutput)
  local v, p = unpack(input)
  p = p[1]
  self.gradInput[1] = self.gradInput[1] or input[1].new()
  self.gradInput[2] = self.gradInput[2] or input[2].new()
  self.gradInput[2]:resizeAs(input[2])

  self.gradInput[1]:set(torch.cmul(gradOutput, torch.pow(v, p - 1)) * p)
  local pgrad = 0
  for i = 1, v:size(1) do
    if v[i] > 0 then
      pgrad = pgrad + math.log(v[i]) * self.output[i] * gradOutput[i]
    end
  end
  self.gradInput[2][1] = pgrad
  return self.gradInput
end
