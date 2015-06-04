--[[
  Copyright 2014 Google Inc. All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
]]--

local Embedding, parent = torch.class('Embedding', 'nn.Module')

function Embedding:__init(inputSize, outputSize)
  parent.__init(self)
  self.outputSize = outputSize
  self.weight = torch.Tensor(inputSize, outputSize)
  self.gradWeight = torch.Tensor(inputSize, outputSize)
end

function Embedding:updateOutput(input)
  self.output:resize(input:size(1), self.outputSize)
  for i = 1, input:size(1) do
    self.output[i]:copy(self.weight[input[i]])
  end
  return self.output
end

function Embedding:updateGradInput(input, gradOutput)
  if self.gradInput then
    self.gradInput:resize(input:size())
    return self.gradInput
  end
end

function Embedding:accGradParameters(input, gradOutput, scale)
  scale = scale or 1
  if scale == 0 then
    self.gradWeight:zero()
  end
  for i = 1, input:size(1) do
    local word = input[i]
    self.gradWeight[word]:add(gradOutput[i])
  end
end

-- we do not need to accumulate parameters when sharing
Embedding.sharedAccUpdateGradParameters = Embedding.accUpdateGradParameters
