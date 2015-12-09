require 'nn'

local TensorTrain, parent = torch.class('nn.TensorTrain', 'nn.Module')

function TensorTrain:__init(outChannels, outHeight, outWidth)
	parent.__init(self)

	self.weight = torch.Tensor()
	self.bias = torch.Tensor()
	self.gradWeight = torch.Tensor()
	self.gradBias = torch.Tensor()

	self.outHeight = outHeight
	self.outWidth = outWidth
	self.outChannels = outChannels

	self.W = {
		n = nil,
		m = nil,
		tt = {core = nil, ps = nil, r = nil},
		mul = function(a, b)
			local n=a.n; local m=a.m; local tt=a.tt; local cra=tt.core; local d=tt.d; local ps=tt.ps; local r=tt.r;
			local rb=b:size(2);
			local c=torch.view(b,torch.cat(m:t(),rb, 1):long());

			for k=1,d do
			  local cr=cra:sub(ps(k),ps[k+1]-1);
			  cr=torch.view(cr,r[k],n[k],m[k],r[k+1]);
			  cr=torch.permute(cr,2,4,1,3); cr=torch.view(cr,n[k]*r[k+1],r[k]*m[k]);
			  local M=c:nElement();
			  c=torch.view(c,r[k]*m[k],M/(r[k]*m[k]));
			  c=cr*c; c=torch.view(c,n[k],c:nElement()/n[k]);
			  c=torch.permute(c,2,1);
			end
			c=c:view(-1); c=torch.view(c,rb,c:nElement()/rb);
			c=c:t();
			return c
		end,

		t = function(tt)
			local t = tt.tt;
			local m = tt.m;
			local n = tt.n;
			local d = t.d;
			local r = t.r;
			for i=1,d do
				local cr = t[i]
				cr = torch.view(cr, r[i], n[i], m[i], r[i+1]);
				cr = torch.permute(cr, 1, 3, 2, 4);
				t[i] = torch.view(cr, r[i], m[i]*n[i], r[i+1]);

			end
			local tt1 = {mul = tt.mul, t = tt.t, rank = tt.rank, tocell = tt.tocell};
			tt1.tt = t;
			tt1.m=tt.n;
			tt1.n=tt.m;
			return tt1
		end,

		rank = function(a)
			return r=a.tt.r;
		end,

		tocell = function(tt)
			local d = tt.tt.d;
			local cc = {}
			local n = tt.n;
			local m = tt.m;
			local r = tt.tt.r;
			local ps = tt.tt.ps;
			local cr = tt.tt.core;
			for i=1:d do
				cc[i] = torch.view(cr:sub(ps(i),(ps(i+1)-1)), r[i], n[i], m[i], r[i+1]);
			end
			return cc
		end
	}

	--TODO: should the constructor arguments resemble Linear or SpatialConvolution?
	--TODO: self:reset()
end

function TensorTrain:updateOutput(input)
	assert(input:dim() == 4)

	local inHeight, inWidth, inChannels, batchSize = input:size(1), input:size(2), input:size(3), input:size(4)

	self.output = W:mul(torch.view(input, -1, batchSize))
	if self.bias:nElement() > 0 then
		self.output:add(torch.view(self.bias, self.outHeight, self.outWidth, self.outChannels, 1):expandAs(self.output))
	end
	self.output = torch.view(self.output, self.outHeight, self.outWidth, self.outChannels, batchSize)
	return self.output
end

function TensorTrain:updateGradInput(input, gradOutput)
	local inHeight, inWidth, inChannels, batchSize = input:size(1), input:size(2), input:size(3), input:size(4)

	self.gradInput = W:t():mul(torch.view(self.gradInput, -1, batchSize))
	self.gradInput = torch.view(self.gradInput, inHeight, inWidth, inChannels, batchSize)
	return self.gradInput
end

function TensorTrain:accGradParameters(input, gradOutput, scale)
	local inHeight, inWidth, inChannels, batchSize = input:size(1), input:size(2), input:size(3), input:size(4)
	if self.bias:nElement() > 0 then
		self.gradBias = self.gradInput:sum(4)
	else
		self.gradBias = []
	end

	local DZDWCore = input.new(W_core:size()):zero()
	local rankArr = self.W:rank()
	local corePos = W.ps

	local numDims = W.n:size(1)
	local coreArr = W:tocell()

	local rightSum = torch.view(input, -1, batchSize)
	rightSum = rightSum:t()

	local leftSum
	for derDim = numDims, 1, -1 do
		if derDim < numDims then
			local rightDim = derDim + 1
			local sumSize = W.m[rightDim] * rankArr[rightDim+1]
			local core = torch.view(coreArr[rightDim], -1, sumSize)
			rightSum = torch.view(rightSum, -1, W.m[rightDim])
			rightSum = core * (torch.view(rightSum:t(), sumSize, -1))
		end

		if derDim >= 2 then
			local core = torch.permute(coreArr[derDim-1], 1, 2, 4, 3)
			core = torch.view(core, -1, W.m[derDim-1])

			leftSum = torch.view(rightSum, rankArr[derDim+1]*torch.prod(W.n:sub(derDim+1, -1))*batchSize*torch.prod(W.m:sub(1, derDim-2)), torch.prod(W.m:sub(derDim-1, derDim)))
        	leftSum = core * torch.view(leftSum:t(), W.m[derDim-1], -1)

			local leftSumDims = torch.LongStorage{rankArr[derDim-1]*W.n[derDim-1], rankArr[derDim]*W.m[derDim]*rankArr[derDim+1], torch.prod(W.n:sub(derDim+1, -1))*batchSize, torch.prod(W.m:sub(1, derDim-2))}
	        leftSum = torch.view(leftSum, leftSumDims)
		    leftSum = torch.permute(leftSum, 1, 3, 2, 4)

			for leftDim = derDim-2:1,-1 do
				local sumSize = W.m[leftDim] * rankArr[leftDim+1]
				core = torch.view(coreArr[leftDim], -1, sumSize)
				leftSum = torch.view(leftSum, -1, W.m[leftDim])
				leftSum = core * torch.view(leftSum:t(), sumSize, -1)
			end
		elseif derDim == 1 then
			leftSum = torch.view(rightSum, rankArr[derDim+1], -1, batchSize, W.m[derDim])
	       	leftSum = torch.permute(leftSum, 2, 3, 4, 1)
		else
			error('Something bad happened(')
		end

		local coreSize = rankArr[derDim] * W.n[derDim] * W.m[derDim] * rankArr[derDim+1]
	    local leftISize = torch.prod(W.n:sub(1, derDim-1))
		local rightISize = torch.prod(W.n:sub(derDim+1, -1))

		local currout_dzdx = torch.view(self.gradInput, leftISize, W.n[derDim], rightISize*batchSize)

		currout_dzdx = torch.permute(currout_dzdx, 2, 1, 3)
		local sumSize = leftISize * rightISize * batchSize
		local der = torch.view(currout_dzdx, -1, sumSize) * torch.view(leftSum, sumSize, -1)

		der = torch.view(der, W.n[derDim], rankArr[derDim], W.m[derDim]*rankArr[derDim+1])
		der = torch.permute(der, 2, 1, 3)
		DZDWCore:sub(corePos[derDim], corePos[derDim+1]-1) = der
	end
	self.gradWeight = DZDWCore
end
