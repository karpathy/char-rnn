
# char-rnn

This code implements **multi-layer Recurrent Neural Network** (RNN, LSTM, and GRU) for training/sampling from character-level language models. The input is a single text file and the model learns to predict the next character in the sequence. 

The context of this code base is described in detail in my [blog post](http://karpathy.github.io/).

There is also a [project page](http://cs.stanford.edu/people/karpathy/char-rnn/) that has some pointers and datasets.

This code is based on Oxford University Machine Learning class [practical 6](https://github.com/oxford-cs-ml-2015/practical6), which is in turn based on [learning to execute](https://github.com/wojciechz/learning_to_execute) code from Wojciech Zaremba. Chunks of it were also developed in collaboration with my labmate [Justin Johnson](https://github.com/jcjohnson/).

## Requirements

This code is written in LUA and requires [Torch](http://torch.ch/).

Additionally, you need to install `nngraph` using [LuaRocks](https://luarocks.org/)

```bash
luarocks install nngraph
```

## Usage


### Data

All input data is stored inside the `data/` directory. You'll notice that there is an example dataset included in the repo (in folder `data/tinyshakespeare`) which consists of a subset of works of Shakespeare. I'm providing a few more datasets on the [project page](http://cs.stanford.edu/people/karpathy/char-rnn/).

**Your own data**: If you'd like to use your own data create a single file `input.txt` and place it into a folder in `data/`. For example, `data/some_folder/input.txt`. The first time you run the training script it will write two more convenience files into `data/some_folder`.

Note that if your data is too small (1MB is already considered very very small) the RNN won't learn very effectively. Remember that it has to learn everything completely from scratch. But if you insist on smaller datasets you might want to decrease the batch size a bit and do many more epochs (hundreds perhaps).

### Training

Start training the model using `train.lua`, for example:

```
$ th train.lua -data_dir data/some_folder -gpuid -1
```

The `-data_dir` flag is most important since it specifies the dataset to use. Notice that in this example we're also setting `gpuid` to -1 which tells the code to train using CPU, otherwise it defaults to GPU 0. There are many other flags for various options. Consult `$ th train.lua -help` for comprehensive settings. Here's another example:

```
$ th train.lua -data_dir data/some_folder -rnn_size 512 -num_layers 2 -dropout 0.5
```

While the model is training it will periodically write checkpoint files to the `cv` folder. You can use these checkpoints to generate text:

### Sampling

Given a checkpoint file (such as those written to `cv`) we can generate new text. For example:

```
$ th sample.lua cv/some_checkpoint.t7 -gpuid -1
```

Make sure that if your checkpoint was trained with GPU it is also sampled from with GPU, or vice versa. Otherwise the code will (currently) complain. As with the train script, see `$ th sample.lua -help` for full options. One important one is (for example) `-length 10000` which would generate 10,000 characters (default = 2000).

**Temperature**. An important parameter you may want to play with a lot is `-temparature`, which takes a number in range (0, 1] (notice 0 not included), default = 1. The temperature is dividing the predicted log probabilities before the Softmax, so lower temperature will cause the model to make more likely, but also more boring and conservative predictions. Higher temperatures cause the model to take more chances and increase diversity of results, but at a cost of more mistakes.

**Priming**. It's also possible to prime the model with some starting text using `-primetext`.

Happy sampling!

## License

MIT