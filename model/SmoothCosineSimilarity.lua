--[[

Input: a table of two inputs {M, k}, where
  M = an n-by-m matrix
  k = an m-dimensional vector

Output: an n-dimensional vector

Each element is an approximation of the cosine similarity between k and the
corresponding row of M. It's an approximation since we add a constant to the
denominator of the cosine similarity function to remove the singularity when
one of the inputs is zero.

--]]

local SmoothCosineSimilarity, parent = torch.class('nn.SmoothCosineSimilarity', 'nn.Module')

function SmoothCosineSimilarity:__init(smoothen)
  parent.__init(self)
  self.gradInput = {}
  self.smooth = smoothen or 1e-3
end

function SmoothCosineSimilarity:updateOutput(input)
  local M, k = unpack(input)
  self.rownorms = torch.cmul(M, M):sum(2):sqrt():view(M:size(1))
  self.knorm = math.sqrt(k:dot(k))
  self.dot = M * k
  self.output:set(torch.cdiv(self.dot, self.rownorms * self.knorm + self.smooth))
  return self.output
end

function SmoothCosineSimilarity:updateGradInput(input, gradOutput)
  local M, k = unpack(input)
  self.gradInput[1] = self.gradInput[1] or input[1].new()
  self.gradInput[2] = self.gradInput[2] or input[2].new()

  -- M gradient
  local rows = M:size(1)
  local Mgrad = self.gradInput[1]
  Mgrad:set(k:repeatTensor(rows, 1))
  for i = 1, rows do
    if self.rownorms[i] > 0 then
      Mgrad[i]:add(-self.output[i] * self.knorm / self.rownorms[i], M[i])
    end
    Mgrad[i]:mul(gradOutput[i] / (self.rownorms[i] * self.knorm + self.smooth))
  end

  -- k gradient
  self.gradInput[2]:set(M:t() * torch.cdiv(gradOutput, self.rownorms * self.knorm + self.smooth))
  if self.knorm > 0 then
    local scale = torch.cmul(self.output, self.rownorms)
      :cdiv(self.rownorms * self.knorm + self.smooth)
      :dot(gradOutput) / self.knorm
    self.gradInput[2]:add(-scale, k)
  end
  return self.gradInput
end
