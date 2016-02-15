# Adding RecurrentJS support to char-rnn

This repo forks [Karpathy's Char-rnn](https://github.com/karpathy/char-rnn) to add a file that allows users to export their models trained in Torch into a JSON format that can be read by [RecurrentJS](https://github.com/karpathy/recurrentjs). This should allow users to train complex models using Torch on their GPUs and deploy them in a simple webpage for everyone to use with their browsers.

The current version I am sharing is very much in development, and while it does produce a JSON file that's in the right ballpark, it doesn't work yet.

To help development, I propose all contributors start by fixing the training set to be the same as that on [Karpathy's demo](http://cs.stanford.edu/people/karpathy/recurrentjs/), which is a collection of Paul Graham's essays. The collection itself is tiny, only ~1400 lines, so this means that we can get a model that works while still being tiny enough as to allow manual inspection. The data is already included in the /data/ folder, but I already trained a model in /examples/PaulGraham.t7 using the same parameters as in the RecurrentJS examples (2 layers of size 20 each). I have already run the conversion script and store the result inside /examples/PaulGrahamModel.json (minified) and /examples/PaulGrahamModel.pretty.json (human-readable). For comparison, I have also run RecurrentJS for a bit and stored the net in /examples/PaulGrahamRecurrentJSModel.pretty.json so that one can visually see the differences.

The objective at this stage is to paste the conversion of the .t7 model into [Karpathy's demo](http://cs.stanford.edu/people/karpathy/recurrentjs/) and have it working immediately.

There are currently some problems that I have identified: 

1. The "size" parameter for the RNN is not the same for the two models (Torch and JS). In Torch, each Linear layer has its own bias, so there are as many bias vectors as there are weight matrices (so 1 per gate per layer for both x and h). In JS, each gate has only 1 bias. We might either change the JS code or we might try to "hack" the format a little bit by adding 1's to the input vectors, but I haven't worked on the maths just yet.

2. The JS code was made to enable training in a browser in reasonable time. This meant that Karpathy added an extra layer before the LSTMs that mapped the vocabulary (of size ~50-100) to a much smaller embedding (the example uses size 5). We won't care about training, so given that we will only do forward propagation we should be able to afford skipping this embedding step, but the JS code expects to find a matrix _Wil_ that contains the weights for the embedding. I have therefore hacked it a bit by setting an embedding size to be the same as the vocabulary size, and setting Wil to be an identity matrix.

3. It's simple to convert a Torch table into JSON using json.encode(), but that does not preserve the order of the elements. LUA tables are inherently unordered when their key is not an integer. If you do use an integer as key that's cool, but then you have to be more careful when building the JSON file. I am still not 100% of the impact of this lack of order, so I am currently leaving this aside assuming that it might work even when not ordered.

I think that if these 3 issues get resolved we should be able to make this work.

Usage:

```
$ th export_to_recurrentjs.lua
```

(remind that every parameter is currently hardcoded)