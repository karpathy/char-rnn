--[[

 Input: A table {x, k} of a vector x and a convolution kernel k.

 Output: Circular convolution of x with k.

 TODO: This module can probably be implemented more efficiently.

--]]

local CircularConvolution, parent = torch.class('nn.CircularConvolution', 'nn.Module')

function CircularConvolution:__init()
  parent.__init(self)
  self.gradInput = {}
end

function rotate_left(input, step)
  local output = input.new():resizeAs(input)
  local size = input:size(1)
  output[{{1, size - step}}] = input[{{step + 1, size}}]
  output[{{size - step + 1, size}}] = input[{{1, step}}]
  return output
end

function rotate_right(input, step)
  local output = input.new():resizeAs(input)
  local size = input:size(1)
  output[{{step + 1, size}}] = input[{{1, size - step}}]
  output[{{1, step}}] = input[{{size - step + 1, size}}]
  return output
end

-- function CircularConvolution:updateOutput_orig(input)
--   local a, b = unpack(input)
--   local size = a:size(1)
--   self.b = b:repeatTensor(1,2)
--   local circ = a.new():resize(size, size)
--   for i = 0, size - 1 do
--     circ[i + 1] = self.b:narrow(2, size - i + 1, size)
--   end
--   self.output:set(torch.mv(circ:t(), a))
--   return self.output
-- end

-- function CircularConvolution:updateGradInput_orig(input, gradOutput)
--   local a, b = unpack(input)
--   local size = a:size(1)
--   for i = 1, 2 do
--     self.gradInput[i] = self.gradInput[i] or input[1].new()
--     self.gradInput[i]:resize(size)
--   end

--   a = a:repeatTensor(1, 2)
--   for i = 0, size - 1 do
--     self.gradInput[1][i + 1] = gradOutput:dot(self.b:narrow(2, size - i + 1, size))
--     self.gradInput[2][i + 1] = gradOutput:dot(a:narrow(2, size - i + 1, size))
--   end
--   return self.gradInput
-- end

function CircularConvolution:updateOutput(input)
  local v, k = unpack(input)
  self.size = v:size(1)
  self.kernel_size = k:size(1)
  self.kernel_shift = math.floor(self.kernel_size / 2)
  self.output = v.new():resize(self.size):zero()
  for i = 1, self.size do
    for j = 1, self.kernel_size do
      local idx = i + self.kernel_shift - j + 1
      if idx < 1 then idx = idx + self.size end
      if idx > self.size then idx = idx - self.size end
      self.output[{{i}}]:add(k[j] * v[idx])
    end
  end
  return self.output
end

function CircularConvolution:updateGradInput(input, gradOutput)
  local v, k = unpack(input)
  self.gradInput[1] = self.gradInput[1] or v.new()
  self.gradInput[2] = self.gradInput[2] or k.new()
  self.gradInput[1]:resize(self.size)
  self.gradInput[2]:resize(self.kernel_size)

  local gradOutput2 = rotate_right(gradOutput:repeatTensor(1, 2):view(2 * self.size), self.kernel_shift)
  for i = 1, self.size do
    self.gradInput[1][i] = k:dot(gradOutput2:narrow(1, i, self.kernel_size))
  end

  local v2 = rotate_left(v:repeatTensor(1, 2):view(2 * self.size), self.kernel_shift + 1)
  for i = 1, self.kernel_size do
    self.gradInput[2][i] = gradOutput:dot(v2:narrow(1, self.size - i + 1, self.size))
  end
  return self.gradInput
end
