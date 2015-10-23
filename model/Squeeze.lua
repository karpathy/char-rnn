local Squeeze, parent = torch.class('nn.Squeeze', 'nn.Module')

function Squeeze:updateOutput(input)
  self.size = input:size()
  self.output = input:squeeze()
  return self.output
end

function Squeeze:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput:view(self.size)
  return self.gradInput
end
