# Automatic Article Generator Documentation

## Idea source
[Karpathy](karpathy.github.io) recently posted an interesting article [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) talking about how effective RNN is when applied to certain area, such as [Image Captioning](http://cs.stanford.edu/people/karpathy/deepimagesent/), and text generation. Please find more detail in his [github page](karpathy.github.io). 

Amazed by this post, we decided to test it out, i.e., to build our own Automatic Article Generator. Luckily Karpathy has his code on [github](https://github.com/karpathy/char-rnn). The strategy we took was analyzing his post while reading his code. 

## Theory
To make such a generator, we have to first build a neural network. Long-Short-Term-Memory (LSTM) is recommended model by Karpathy. It is a particular type of RNN, that works better practically, according to him. So we chose that. 

Here we won't go too much into the detail of how LSTM works. Two post from [Adam Paszka](https://apaszke.github.io/lstm-explained.html) and [Christopher Olah](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) well explained the structure of a typical one-layer LSTM (shown below, by Chris).
![Alt text](img/one_layer_LSTM.png)

Basically they have three gates to filter the information:
- forget gate: decide if to use old information
- input gate: decide if to use new information
- output gate: decide if output transformed cell state


## Coding
###Torch7
We used [torch7](https://github.com/torch/torch7) from facebook to build our LSTM model and execute forward propagation and backward propagation for you with high efficiency. 

The installation is fairly easy, follow the [guide](http://torch.ch/docs/getting-started.html#_). Make sure before running, activate the torch environment by running`. ~/torch/install/bin/torch-activate` in terminal.

Since torch is written in a language called LUA, for this project we use LUA as well, here is a useful LUA [cheat sheet](http://tylerneylon.com/a/learn-lua/). But in the future, we want to use python to organize whole project, only use LUA for computationally intensive module.

###Software structure
Our project has 6 parts, it's really beneficial to read through the structure for people who want to apply this or follow up this work.

- data/: a folder stores input article data, e.g., Shakespeare.txt
	- vocab.t7: a dictionary mapping character to a number
	- data.t7: coded input article using the dictionary mapping
- model/: a folder stores all the model files. In our case, we have a model file `LTSM.lua`. To understand this file, you should 
	- first understand the terminology of LSTM by reading [Christopher Olah's post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
	- read [Adam Paszka](https://apaszke.github.io/lstm-explained.html) and his [one-layer](https://apaszke.github.io/assets/posts/lstm-explained/LSTM.lua) LSTM model code, it could also help you understand the syntax of LUA and torch

- util/: a folder stores all the utility method files. We have
	- loader: 
		- preprocess input article by translating it into coded input
		- divide data into training, validation, and testing
		- prepare data for batch run
	- onehot:
		- convert coded input into a vector where one dimension is 1 and others are zero

- train.lua: basically organize the whole process of training
	- read user input to create instance of LSTM model
	- train using `feval` methoc
	- evaluate loss using validation data by `eval_split(2)`
	- save checkpoints
- checkpoints/: dump all the data at a particular training time point to a checkpoint file

- sample.lua: use trained model to make predictions.
