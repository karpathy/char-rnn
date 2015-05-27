
# char-rnn

This code implements **multi-layer Recurrent Neural Network** (RNN, LSTM, and GRU) for training/sampling from character-level language models. The input is a single text file and the model learns to predict the next character in the sequence. 

The context of this code base is described in detail in my [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

There is also a [project page](http://cs.stanford.edu/people/karpathy/char-rnn/) that has some pointers and datasets.

This code is based on Oxford University Machine Learning class [practical 6](https://github.com/oxford-cs-ml-2015/practical6), which is in turn based on [learning to execute](https://github.com/wojciechz/learning_to_execute) code from Wojciech Zaremba. Chunks of it were also developed in collaboration with my labmate [Justin Johnson](https://github.com/jcjohnson/).

## Requirements

This code is written in Lua and requires [Torch](http://torch.ch/).
Additionally, you need to install the `nngraph` and `optim` packages using [LuaRocks](https://luarocks.org/) which you will be able to do after installing Torch

```bash
$ luarocks install nngraph 
$ luarocks install optim
```

## Usage


### Data

All input data is stored inside the `data/` directory. You'll notice that there is an example dataset included in the repo (in folder `data/tinyshakespeare`) which consists of a subset of works of Shakespeare. I'm providing a few more datasets on the [project page](http://cs.stanford.edu/people/karpathy/char-rnn/).

**Your own data**: If you'd like to use your own data create a single file `input.txt` and place it into a folder in `data/`. For example, `data/some_folder/input.txt`. The first time you run the training script it will write two more convenience files into `data/some_folder`.

Note that if your data is too small (1MB is already considered very small) the RNN won't learn very effectively. Remember that it has to learn everything completely from scratch.

### Training

Start training the model using `train.lua`, for example:

```
$ th train.lua -data_dir data/some_folder -gpuid -1
```

The `-data_dir` flag is most important since it specifies the dataset to use. Notice that in this example we're also setting `gpuid` to -1 which tells the code to train using CPU, otherwise it defaults to GPU 0. There are many other flags for various options. Consult `$ th train.lua -help` for comprehensive settings. Here's another example:

```
$ th train.lua -data_dir data/some_folder -rnn_size 512 -num_layers 2 -dropout 0.5
```

While the model is training it will periodically write checkpoint files to the `cv` folder. The frequency with which these checkpoints are written is controlled with number of iterations, as specified with the `eval_val_every` option (e.g. if this is 1 then a checkpoint is written every iteration).

We can use these checkpoints to generate text (discussed next).

### Sampling

Given a checkpoint file (such as those written to `cv`) we can generate new text. For example:

```
$ th sample.lua cv/some_checkpoint.t7 -gpuid -1
```

Make sure that if your checkpoint was trained with GPU it is also sampled from with GPU, or vice versa. Otherwise the code will (currently) complain. As with the train script, see `$ th sample.lua -help` for full options. One important one is (for example) `-length 10000` which would generate 10,000 characters (default = 2000).

**Temperature**. An important parameter you may want to play with a lot is `-temparature`, which takes a number in range (0, 1] (notice 0 not included), default = 1. The temperature is dividing the predicted log probabilities before the Softmax, so lower temperature will cause the model to make more likely, but also more boring and conservative predictions. Higher temperatures cause the model to take more chances and increase diversity of results, but at a cost of more mistakes.

**Priming**. It's also possible to prime the model with some starting text using `-primetext`.

Happy sampling!

## Tips and Tricks

### Monitoring Validation Loss vs. Training Loss
If you're somewhat new to Machine Learning or Neural Networks it can take a bit of expertise to get good models. The most important quantity to keep track of is the difference between your training loss (printed during training) and the validation loss (printed once in a while when the RNN is run on the validation data (by default every 1000 iterations)). In particular:

- If your training loss is much lower than validation loss then this means the network is **overfitting**. Solutions to this are to decrease your network size, or to increase dropout. For example you could try dropout of 0.5 and so on.
- If your training/validation loss are about equal then your model is **underfitting**. Increase the size of your model (either number of layers or the raw number of neurons per layer)

### Approximate number of parameters

The two most important parameters that control the model are `rnn_size` and `num_layers`. I would advise that you always use `num_layers` of about 3. The `rnn_size` can be adjusted based on how much data you have. The two important quantities to keep track of here are:

- The number of parameters in your model. This is printed when you start training.
- The size of your dataset. 1MB file is approximately 1 million characters.

These two should be about the same order of magnitude. It's a little tricky to tell. Here are some examples:

- I have a 100MB dataset and I'm using the default parameter settings (which currently print 150K parameters). My data size is significantly larger (100 mil >> 0.15 mil), so I expect to heavily underfit. I am thinking I can comfortably afford to make `rnn_size` larger.
- I have a 10MB dataset and running a 10 million parameter model. I'm slightly nervous and I'm carefully monitoring my validation loss. If it's larger than my training loss then I may want to increase dropout a bit.

### Best models strategy

The winning strategy to obtaining very good models (if you have the compute time) is to always err on making the network larger (as large as you're willing to wait for it to compute) and then try different dropout values (between 0,1). Whatever model has the best validation performance (the loss, written in the checkpoint filename, low is good) is the one you should use in the end.

It is very common in deep learning to run many different models with many different hyperparameter settings, and in the end take whatever checkpoint gave the best validation performance.

By the way, the size of your training and validation splits are also parameters. Make sure you have a decent amount of data in your validation set or otherwise the validation performance will be noisy and not very informative.

## License

MIT
