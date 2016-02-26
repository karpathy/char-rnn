--[[
This file reads a model trained using train.lua in this repo
and exports it in a JSON format compatible with RecurrentJS
so that it can be run by a browser.




]] --

-- simple script that loads a checkpoint and prints its opts
--require('mobdebug').start()  -- Uncomment this line if you want to debug in terminal or in zbs-studio
require 'torch'
require 'nn'
require 'nngraph'
require 'io'
require 'util.OneHot'
require 'util.misc'
local ok, cunn = pcall(require, 'cunn')
local ok2, cutorch = pcall(require, 'cutorch')

function createWeightsTable(cudaTensor)
    local thistable = {}
    doubleTensor = cudaTensor:double()
    thistable.n = doubleTensor:size(1)
    thistable.d = doubleTensor:size(2)
    thistable.w = {}
    for i = 1, doubleTensor:size(1) do
        for j = 1,doubleTensor:size(2) do
            indexInt = ((i-1) * doubleTensor:size(2)) + j - 1
            indexStr = tostring(indexInt)
            thistable.w[indexStr] = doubleTensor[i][j]
      end
    end  
    return thistable
end

function createBiasTable(cudaTensor)
    local thistable = {}
    doubleTensor = cudaTensor:double()
    thistable.n = doubleTensor:size(1)
    thistable.d = 1
    thistable.w = {}
    for i = 1, doubleTensor:size(1) do
          thistable.w[i-1] = doubleTensor[i]
    end  
    return thistable
  
end

-- This also makes the vocab 0-indexed as RecurrentJS would have it.
function escapeVocab(vocab) 
  escapedvocab = {}
  local inspect = require 'inspect'
  for key, val in pairs(vocab) do
      escapedkey = fixUTF8(inspect(key), "Invalid")
      if (not string.find(escapedkey, "Invalid")) then
        escapedvocab[escapedkey] = val -1
      end
  end
  return escapedvocab
end

function invertTable(vocab)
    t = {}
    for k, v in pairs(vocab) do
      escapedk = string.sub(k, 2, #k-1)
      strval = tostring(v)
      t[strval] = escapedk
    end
  return t
end

function getKeys(vocab)
  t = {}
  for k, v in pairs(vocab) do
    table.insert(t, k)
  end
  return t
end

function fixUTF8(s, replacement)
  local p, len, invalid = 1, #s, {}
  while p <= len do
    if     p == s:find("[%z\1-\127]", p) then p = p + 1
    elseif p == s:find("[\194-\223][\128-\191]", p) then p = p + 2
    elseif p == s:find(       "\224[\160-\191][\128-\191]", p)
        or p == s:find("[\225-\236][\128-\191][\128-\191]", p)
        or p == s:find(       "\237[\128-\159][\128-\191]", p)
        or p == s:find("[\238-\239][\128-\191][\128-\191]", p) then p = p + 3
    elseif p == s:find(       "\240[\144-\191][\128-\191][\128-\191]", p)
        or p == s:find("[\241-\243][\128-\191][\128-\191][\128-\191]", p)
        or p == s:find(       "\244[\128-\143][\128-\191][\128-\191]", p) then p = p + 4
    else
      s = s:sub(1, p-1)..replacement..s:sub(p+1)
      table.insert(invalid, p)
    end
  end
  return s, invalid
end

-- Yeah, turns out LUA needs this
function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

json = require("json")

path = "examples/PaulGraham128"

local model = torch.load(path .. ".t7") -- Given we are still doing development, this is currently fixed

rnn = model.protos.rnn

AllModelWeights = {}

-- The way weights are stored is simple:
-- Each layer has 2 nn.Linear() layers with the weights from input to hidden
-- and hidden to hidden, respectively. Each of these 2 matrices has size
-- [layer_input, 4 * rnn_size]. Layer_input is of size [Voc] for the first layer,
-- and of size rnn_size for all other layers.
-- The quadruple size comes from the fact that 4 matrices are concatenated there.
-- In order, these are the weights for: 
--    - input gate, 
--    - forget gate, 
--    - output gate, 
--    - new memory cell

Biases = {}

-------------- The following region handles weight matrices
local LinearModules = rnn:findModules("nn.Linear")

for i,linearmodule in ipairs(LinearModules) do
        if (i == 1) then
          AllModelWeights["Wil"] = createWeightsTable(torch.eye(linearmodule.weight:size(2)))
        end -- if n == 1 we add this, but also add x0
          if (i == #LinearModules) then
            AllModelWeights["Whd"] = createWeightsTable(linearmodule.weight)
            Biases["bd"] = linearmodule.bias
  
          else
            local W = torch.reshape(linearmodule.weight, 4, linearmodule.weight:size(1) / 4, linearmodule.weight:size(2)) -- These are all packed and need unpacking into i, f, o, g. The last gate, g, is called c in the new format.
            local B = torch.reshape(linearmodule.bias, 4, linearmodule.bias:size(1) / 4)
            if(i % 2 == 1) then 
              AllModelWeights["Wix" .. (i-1)/2] = createWeightsTable(W[1])
              Biases["bix".. (i-1)/2] = B[1]
              
              AllModelWeights["Wfx" .. (i-1)/2] = createWeightsTable(W[2])
              Biases["bfx".. (i-1)/2] = B[2]

              AllModelWeights["Wox" .. (i-1)/2] = createWeightsTable(W[3])
              Biases["box".. (i-1)/2] = B[3]

              AllModelWeights["Wcx" .. (i-1)/2] = createWeightsTable(W[4])
              Biases["bcx".. (i-1)/2] = B[4]
              
              print("Processed Wx" .. (i-1)/2)
            else 
              AllModelWeights["Wih" .. math.floor((i-1)/2)] = createWeightsTable(W[1])
              Biases["bih" .. math.floor((i-1)/2)] = B[1]

              AllModelWeights["Wfh" .. math.floor((i-1)/2)] = createWeightsTable(W[2])
              Biases["bfh" .. math.floor((i-1)/2)] = B[2]

              AllModelWeights["Woh" .. math.floor((i-1)/2)] = createWeightsTable(W[3])
              Biases["boh" .. math.floor((i-1)/2)] = B[3]

              AllModelWeights["Wch" .. math.floor((i-1)/2)] = createWeightsTable(W[4])
              Biases["bch" .. math.floor((i-1)/2)] = B[4]              
              
              print("Processed Wh" .. math.floor((i-1)/2))
            end
          end
      
end


-----------------------------------------------------------------
--The following region handles biases (for each gate, we have to sum up the contribution coming from x with that coming from h)
--Some printing (leaving it here to help development)
for i=0,model.opt.num_layers-1 do -- Recall that RecurrentJS is 0-indexed
  AllModelWeights["bi" .. i] = createBiasTable( Biases["bix" .. i] + Biases["bih" .. i] ) 
  AllModelWeights["bf" .. i] = createBiasTable( Biases["bfx" .. i] + Biases["bfh" .. i] )
  AllModelWeights["bo" .. i] = createBiasTable( Biases["box" .. i] + Biases["boh" .. i] )
  AllModelWeights["bc" .. i] = createBiasTable( Biases["bcx" .. i] + Biases["bch" .. i] )
  AllModelWeights["bd"] = createBiasTable(Biases["bd"])
end


-----------------------------------------------------------------------
print("Processed biases")

--print(AllModelWeights)

-- ConvNetJS is more flexible than this Torch code because it allows different layers to be of different size.
-- This does not support it, so we just iterate through each layer and copy the same layer size in the JSON.
hiddenSizes = {}
for i = 1, model.opt.num_layers do
  table.insert(hiddenSizes, model.opt.rnn_size)
end

fho,err = io.open(path .. ".json","w")

modelstr = json.encode(AllModelWeights)
modelstr = modelstr:gsub("\\[", ""):gsub("\\]", "")

vocab = escapeVocab(model.vocab)

mymodel = {}

mymodel.generator       = model.opt.model
mymodel.model           = AllModelWeights
mymodel.i               = model.i
mymodel.letterToIndex   = vocab
mymodel.indexToLetter   = invertTable(vocab)
mymodel.vocab           = getKeys(vocab)
mymodel.hidden_sizes    = hiddenSizes
mymodel.letter_size     = tablelength(model.vocab) -- Size of the embeddings for RecurrentJS smaller than the vocab in input. For us it's not.
mymodel.solver          = {}
mymodel.solver["decay_rate"] = 0.999
mymodel.solver["smooth_eps"] = 1E-8

mymodelStr = json.encode(mymodel):gsub("\\\"", ""):gsub("\\'", "")
fho:write(mymodelStr)
fho:flush()
fho:close()

print("Done!")









