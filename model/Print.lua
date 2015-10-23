require "gnuplot"
--[[
 An Identity layer that prints its input.
--]]
local Print, parent = torch.class('nn.Print', 'nn.Module')
function Print:__init(label)
  parent:__init(self)
  self.label = label
end
function Print:updateOutput(input)
  self.output = input
  if self.label ~= nil then
    print(self.label)
  end
  print(input)
  return self.output
end
function Print:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput
  return self.gradInput
end

--print tensor size
local PrintSize, parent = torch.class('nn.PrintSize', 'nn.Module')
function PrintSize:__init(label)
  parent:__init(self)
  self.label = label
end
function PrintSize:updateOutput(input)
  self.output = input
  local sizes = {}
  local size = input:size()
  for i=1,input:nDimension() do
    table.insert(sizes, size[i])
  end
  print(string.format("%s -> %s", self.label,table.concat(sizes,"x")))
  return self.output
end
function PrintSize:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput
  return self.gradInput
end



local PrintAddress, parent = torch.class('nn.PrintAddress', 'nn.Module')

function PrintAddress:__init(label)
  parent:__init(self)
  self.label = label
  self.look = 0
end

function PrintAddress:updateOutput(input)
  self.output = input
  local v, index = torch.max(input[1],1)
  v, index = v[1], index[1]
  if self.look ~= index then
    print(string.format('%s moved from %d to %d %.2f',self.label,self.look,index,v))
    self.look = index
  end
  return self.output
end


function PrintAddress:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput
  return self.gradInput
end

local PrintTensor, parent = torch.class('nn.PrintTensor', 'nn.Module')

function PrintTensor:__init(interval, label)
  parent:__init(self)
  self.label = label
  self.interval = interval
  self.count = 0
end

function PrintTensor:updateOutput(input)
  self.output = input
  --gnuplot.figure(1)
  self.count = self.count + 1
  if self.count % self.interval == 0 then
    gnuplot.title(self.label)
    if input:nDimension() == 3 then
      gnuplot.imagesc(input[1],'memory')
    else
      gnuplot.imagesc(input ,'memory')
    end
  end
  return self.output
end


function PrintTensor:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput
  return self.gradInput
end
