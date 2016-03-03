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
    doubleTensor = cudaTensor:float()
    thistable.n = doubleTensor:size(1)
    thistable.d = doubleTensor:size(2)
    thistable.w = {}
    for i = 1, doubleTensor:size(1) do
        for j = 1,doubleTensor:size(2) do
            indexInt = ((i-1) * doubleTensor:size(2)) + j - 1
            indexStr = tostring(indexInt)
            -- Truncates Weight to 5 digits, simulating a FP16 conversion
            truncatedWeight = tonumber(string.format("%." .. (5) .. "f", doubleTensor[i][j]))
            thistable.w[indexStr] = truncatedWeight
      end
    end  
    return thistable
end

function createBiasTable(cudaTensor)
    local thistable = {}
    doubleTensor = cudaTensor:float()
    thistable.n = doubleTensor:size(1)
    thistable.d = 1
    thistable.w = {}
    for i = 1, doubleTensor:size(1) do
          truncatedWeight = tonumber(string.format("%." .. (5) .. "f", doubleTensor[i]))
          thistable.w[i-1] = doubleTensor[i]
    end  
    return thistable
  
end

-- This function escapes each entry and creates a 0-indexed vocabulary
function escapeVocab(vocab) 
  escapedvocab = {}
  local inspect = require 'inspect'
  for key, val in pairs(vocab) do
      escapedkey = fixUTF8(inspect(key), "Invalid")
      if (not string.find(escapedkey, "Invalid")) then
        escapedvocab[escapedkey] = val - 1 -- making it 0-indexed
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


if (#arg < 1) then
  io.write("No path given as argument. Type path here: ")
  path = io.read()
else 
  path = arg[1]
end

local model = torch.load(path)

rnn = model.protos.rnn


-- json.encode would work well if we put all the weights in a table and then encoded it.
-- Unfortunately, LUAJit objects can't be bigger than 1 GB even in x64 systems
-- so we need to be creative and stream vectors in output instead, generating the
-- JSON on the fly.
-- This function helps do just that.
function streamWriteWeightsTable(fileDescriptor, tableName, table)
  fileDescriptor:write('"'.. tableName .. '":')
  fileDescriptor:write(json.encode(table))
end


-- ConvNetJS is actually more flexible than char-rnn because 
-- it allows different layers to be of different size.
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
--mymodel.model           = AllModelWeights
mymodel.letterToIndex   = vocab
mymodel.indexToLetter   = invertTable(vocab)
mymodel.vocab           = getKeys(vocab)
mymodel.hidden_sizes    = hiddenSizes
mymodel.letter_size     = tablelength(model.vocab) -- Size of the embeddings for RecurrentJS smaller than the vocab in input. For us it's not.
mymodel.solver          = {}
mymodel.solver["decay_rate"] = 0.999 -- RecurrentJS needs these, even though most people won't be training there anyway.
mymodel.solver["smooth_eps"] = 1E-8

mymodelStr = json.encode(mymodel):gsub("\\\"", ""):gsub("\\'", "")
mymodelStr = mymodelStr:sub(1, #mymodelStr-1) -- Remove last closing bracket
mymodelStr = mymodelStr .. ',"model":{'
fho:write(mymodelStr)


Biases = {} -- We can afford to store biases

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

-----------------------------------------------------------------------
-------------- The following region handles weight matrices
local LinearModules = rnn:findModules("nn.Linear")

for i,linearmodule in ipairs(LinearModules) do
        if (i == 1) then
          streamWriteWeightsTable(fho, "Wil", createWeightsTable(torch.eye(linearmodule.weight:size(2))) )
          fho:write(',')
        end -- if n == 1 we add this, but also add x0
          if (i == #LinearModules) then
            streamWriteWeightsTable(fho, "Whd", createWeightsTable(linearmodule.weight) )
            fho:write(',')
            Biases["bd"] = linearmodule.bias
  
          else
            local W = torch.reshape(linearmodule.weight, 4, linearmodule.weight:size(1) / 4, linearmodule.weight:size(2)) -- These are all packed and need unpacking into i, f, o, g. The last gate, g, is called c in the new format.
            local B = torch.reshape(linearmodule.bias, 4, linearmodule.bias:size(1) / 4)
            if(i % 2 == 1) then
              streamWriteWeightsTable(fho, "Wix" .. (i-1)/2, createWeightsTable(W[1]) )
              fho:write(',')
              Biases["bix".. (i-1)/2] = B[1]
              
              streamWriteWeightsTable(fho, "Wfx" .. (i-1)/2, createWeightsTable(W[2]) )
              fho:write(',')
              Biases["bfx".. (i-1)/2] = B[2]
  
              streamWriteWeightsTable(fho, "Wox" .. (i-1)/2, createWeightsTable(W[3]) )
              fho:write(',')
              Biases["box".. (i-1)/2] = B[3]

              streamWriteWeightsTable(fho, "Wcx" .. (i-1)/2, createWeightsTable(W[4]) )
              fho:write(',')
              Biases["bcx".. (i-1)/2] = B[4]
              
              print("Processed x" .. (i-1)/2)
            else 
              streamWriteWeightsTable(fho, "Wih" .. math.floor((i-1)/2), createWeightsTable(W[1]) )
              fho:write(',')
              Biases["bih" .. math.floor((i-1)/2)] = B[1]

              streamWriteWeightsTable(fho, "Wfh" .. math.floor((i-1)/2), createWeightsTable(W[2]) )
              fho:write(',')
              Biases["bfh" .. math.floor((i-1)/2)] = B[2]

              streamWriteWeightsTable(fho, "Woh" .. math.floor((i-1)/2), createWeightsTable(W[3]) )
              fho:write(',')
              Biases["boh" .. math.floor((i-1)/2)] = B[3]

              streamWriteWeightsTable(fho, "Wch" .. math.floor((i-1)/2), createWeightsTable(W[4]) )
              fho:write(',')
              Biases["bch" .. math.floor((i-1)/2)] = B[4]              
              
              print("Processed h" .. math.floor((i-1)/2))
            end
          end
      
end

-----------------------------------------------------------------
--The following region handles biases (for each gate, we have to sum up the contribution coming from x 
-- with that coming from h due to the different format ConvNetJS uses)
--Some printing (leaving it here to help development)
for i=0,model.opt.num_layers-1 do -- Recall that RecurrentJS is 0-indexed
  streamWriteWeightsTable(fho, "bi" .. i, createBiasTable( Biases["bix" .. i] + Biases["bih" .. i] ) )
  fho:write(',')
  streamWriteWeightsTable(fho, "bf" .. i, createBiasTable( Biases["bfx" .. i] + Biases["bfh" .. i] ) )
  fho:write(',')
  streamWriteWeightsTable(fho, "bo" .. i, createBiasTable( Biases["box" .. i] + Biases["boh" .. i] ) )
  fho:write(',')
  streamWriteWeightsTable(fho, "bc" .. i, createBiasTable( Biases["bcx" .. i] + Biases["bch" .. i] ) )
  fho:write(',')
end

  streamWriteWeightsTable(fho, "bd", createBiasTable(Biases["bd"]) )
  -- No commas for the last one!



fho:write("}}")
fho:flush()
fho:close()

print("Done! Converted file is in " .. path .. ".json")






