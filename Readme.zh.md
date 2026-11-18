# char-rnn

[English](Readme.md) | [简体中文](Readme.zh.md)

本项目基于 Torch 实现了用于字符级语言模型训练与采样的**多层循环神经网络**（Multi-layer Recurrent Neural Network，支持 RNN、LSTM 与 GRU）。换言之，模型接收单个纯文本文件作为输入，训练循环神经网络学习在给定上下文序列的前提下预测下一个字符。训练完成的 RNN 随后可逐字符自主生成具备与原始训练语料相同风格与结构的全新文本。关于本代码库的背景与详细技术语境，可参阅 Andrej Karpathy 的经典技术博文[《循环神经网络的不合理有效性》（The Unreasonable Effectiveness of Recurrent Neural Networks）](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)。

对于初次接触 Torch、Lua 或神经网络的开发者，本代码本质上是作者此前编写的 [100 行 Python/NumPy 极简参考实现](https://gist.github.com/karpathy/d4dee566867f8291f086) 的进阶版。本仓库在此基础上提供了诸多拓展功能：支持堆叠多层网络、默认采用表达能力更强的 LSTM 代替经典标准 RNN、提供完备的模型检查点（Checkpoint）存取机制，并且原生基于 mini-batch 批处理与 GPU 加速，计算效率大幅提升。

## 更新：推荐使用现代重构版本 torch-rnn

[Justin Johnson](http://cs.stanford.edu/people/jcjohns/)（@jcjohnson）此前基于 Torch 从零重构了 char-rnn，打造了代码更精炼、结构更优雅、运行更高效的全新代码库 [torch-rnn](https://github.com/jcjohnson/torch-rnn)。该重构版本默认采用 Adam 优化器，并在底层硬编码实现了空间与时间效率更优的 RNN/LSTM 前向与反向传播，彻底规避了本仓库中模型深拷贝（Clone）的复杂机制。对于当前的实际工程需求，推荐优先使用 torch-rnn 作为首选基准实现。

## 环境依赖 (Requirements)

本代码采用 Lua 语言编写，依赖 [Torch](http://torch.ch/) 科学计算框架。在 Ubuntu 环境下，在用户主目录安装 Torch 的标准命令如下：

```bash
$ curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
$ git clone https://github.com/torch/distro.git ~/torch --recursive
$ cd ~/torch;
$ ./install.sh      # 并在末尾输入 "yes" 以更新 .bashrc 环境变量
$ source ~/.bashrc
```

更多安装细节请参阅 Torch 官方安装文档。Torch 安装就绪后，需通过随附的 [LuaRocks](https://luarocks.org/) 包管理器安装核心依赖扩展包：

```bash
$ luarocks install nngraph
$ luarocks install optim
$ luarocks install nn
```

若计划利用 NVIDIA GPU 进行 CUDA 硬件加速训练（相比 CPU 通常可提速约 15 倍），需确保系统已安装适配的 [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)，随后安装 `cutorch` 与 `cunn` 包：

```bash
$ luarocks install cutorch
$ luarocks install cunn
```

若希望在 AMD/ATI 等支持 OpenCL 的显卡上进行 GPU 加速，需安装 `cltorch` 与 `clnn` 包，并在训练时指定 `-opencl 1` 参数（相关已知问题详见 [cltorch issues](https://github.com/hughperkins/cltorch/issues)）：

```bash
$ luarocks install cltorch
$ luarocks install clnn
```

## 使用指南 (Usage)

### 数据集准备

所有输入语料均存放在 `data/` 目录中。仓库内自带示例数据集 `data/tinyshakespeare`（莎士比亚作品子集）。[作者主页](http://cs.stanford.edu/people/karpathy/char-rnn/) 还提供了更多精选训练数据集供下载探索。

**使用自定义数据**：如需使用专属语料，只需创建名为 `input.txt` 的纯文本文件，并将其放入 `data/` 下的新建子目录中（例如 `data/some_folder/input.txt`）。在首次运行训练脚本时，程序会自动进行预处理分词，并在该子目录下生成两个轻量缓存文件。

**数据集规模建议**：若训练语料过小（1MB 已属于极小规模），RNN 很难学到深层规律（模型必须从零开始学习字符组合与语法规则）。反之，当数据规模较大（超过 2MB）时，可放心地增大 `rnn_size` 并训练规模更大的网络模型（详见下文训练说明），模型效果将获得**显著提升**。例如拥有 6MB 数据时，可轻松将 `rnn_size` 设为 300 甚至更高。在显存允许的条件下，本代码库验证过的最大单卡网络配置为 `rnn_size 700` 配合 `num_layers 3`（默认层数为 2）。

### 模型训练 (Training)

通过 `train.lua` 启动模型训练。在内置的示例数据集上快速验证运行的命令如下：

```bash
$ th train.lua -gpuid -1
```

注意此处显式设置了 `-gpuid -1` 参数，强制程序在 CPU 模式下运行；若不设置该参数，程序默认优先使用 GPU 0。系统支持丰富的训练超参数调节，可执行 `$ th train.lua -help` 查阅完整参数矩阵。以下示例展示了如何在自定义数据集上训练规模更大的多层网络（前提是 `data/some_folder/input.txt` 文件已就绪）：

```bash
$ th train.lua -data_dir data/some_folder -rnn_size 512 -num_layers 2 -dropout 0.5
```

**检查点（Checkpoints）**：模型在训练过程中会定期向 `cv/` 目录写入权重检查点文件。保存频率由迭代步数参数 `eval_val_every` 控制（例如设为 1 则每步均写入）。检查点文件名包含一个核心衡量指标：**损失值（Loss）**。例如文件名 `lm_lstm_epoch0.95_2.0681.t7` 表示模型当前训练至第 0.95 个 Epoch（接近完成一轮全量数据遍历），在验证集上的损失值为 2.0681。验证集损失值越低代表模型泛化性能越好。在后续生成文本时，应优先选用验证集损失最低的检查点文件。注意：由于后期可能出现过拟合，最佳检查点不一定对应训练结束时的最后一个检查点。

其他关键超参数包括批大小 `batch_size`（记为 $B$）、序列长度 `seq_length`（记为 $S$）以及数据集划分比例 `train_frac` 和 `val_frac`。批大小决定了并行处理的数据流数量；序列长度决定了单条数据流的展开步长，也是时间反向传播（BPTT）截断的最大梯度的回传深度。例如当 `seq_length` 为 20 时，梯度回传步数上限为 20，模型将无法捕获跨度超过 20 个字符的长程依赖关系。因此对于包含复杂长程依赖的语料，需适当增大该参数。运行时包含 $N$ 个字符的文本首先被切分为大小为 $B \times S$ 的数据块，并按比例划分为训练集/验证集/测试集。默认情况下 `train_frac` 为 0.95，`val_frac` 为 0.05（即 95% 数据用于训练，5% 数据用于估算验证损失）。若数据量较小且参数设置不当，切分后的总块数可能过少（例如仅有 100 块），此时应适当降低批大小或序列长度。

此外，支持通过 `init_from` 参数从先前保存的检查点初始化网络权重并继续微调。

### 文本采样与生成 (Sampling)

利用训练好的检查点权重文件（位于 `cv/` 目录），可通过 `sample.lua` 自主生成全新文本：

```bash
$ th sample.lua cv/some_checkpoint.t7 -gpuid -1
```

注意：若检查点是在 GPU 模式下训练生成的，则采样时亦需在 GPU 模式下运行；反之亦然。可运行 `$ th sample.lua -help` 查看采样脚本的完整选项列表，例如 `-length 10000` 表示连续生成 10,000 个字符（默认值为 2000）。

**采样温度（Temperature）**：`-temperature` 是控制生成多样性的核心调节参数，取值范围为 $(0, 1]$（不包含 0，默认值为 1）。温度参数在 Softmax 之前对对数概率（Logits）进行缩放：较低的温度会使概率分布更尖锐，生成结果更符合高频规律、保守且确定，但可能略显单调；较高的温度会使分布更平滑，显著提升生成内容的多样性与创意度，但产生拼写错误的概率也会随之增加。

**提示前缀（Priming）**：可通过 `-primetext` 为模型提供初始引导文本，使 RNN 在预设的上下文状态下进行“预热”，随后在此基础上继续向后接续生成文本。例如指定 `-primetext "the meaning of life is "`。

**GPU 训练检查点转 CPU 采样**：若模型在 GPU 上完成训练但需在纯 CPU 环境下采样，可使用 `convert_gpu_cpu_checkpoint.lua` 脚本将 GPU 检查点转换为 CPU 兼容权重：

```bash
$ th convert_gpu_cpu_checkpoint.lua cv/lm_lstm_epoch30.00_1.3950.t7
```

转换完成后将生成 `cv/lm_lstm_epoch30.00_1.3950.t7_cpu.t7` 文件，随后即可搭配 `sample.lua` 与 `-gpuid -1` 参数在 CPU 上流畅运行推理。

祝采样顺利！

## 调优技巧与实践经验 (Tips and Tricks)

### 验证集损失 vs 训练集损失监控

对于深度学习初学者，获得优秀的模型需要一定的调优直觉。训练过程中最关键的监控指标是训练损失（每步实时打印）与验证损失（定期在验证集上评估打印，默认每 1000 次迭代）之间的动态差距：

- 若**训练损失远低于验证损失**，表明网络模型出现了**过拟合（Overfitting）**。解决方案包括：缩减网络规模（减小 `rnn_size` 或 `num_layers`），或者增大正则化强度（例如调高 Dropout 至 0.5 等）。
- 若**训练损失与验证损失相近且居高不下**，表明网络模型处于**欠拟合（Underfitting）**状态。解决方案包括：扩充模型容量（增加网络层数 `num_layers` 或每层的隐藏单元数 `rnn_size`）。

### 参数量与数据规模的大致匹配

控制模型容量的两个最关键超参数是 `rnn_size` 与 `num_layers`。工程实践中通常建议设置 `num_layers` 为 2 或 3，并根据语料规模动态调整 `rnn_size`。需权衡比对的两个量为：

- 模型的总参数量（在训练启动时会打印输出）；
- 数据集的大小（1MB 纯文本文件约包含 100 万个字符）。

在理想调优状态下，模型参数量与数据字符规模应处于大致相仿的数量级区间：

- 例如：拥有 100MB 语料库且使用默认配置（仅约 15 万参数），此时数据规模显著大于参数量（1 亿字符 $\gg$ 15 万参数），模型势必严重欠拟合，可以从容大幅增大 `rnn_size`；
- 又如：拥有 10MB 语料库但运行 1000 万参数模型，参数量与数据量相对逼近，此时需高度警惕并密切监控验证损失；若验证损失显著攀升并脱离训练损失，应及时增大 Dropout 抑制过拟合。

### 优胜模型遴选策略

若算力充裕，行之有效的模型调优策略是：优先构建在等待时间内能够完成训练的较大规模网络架构，随后尝试多组不同的 Dropout 取值（介于 0 到 1 之间）。最终以检查点文件名中记录的最低验证集损失（Validation Loss）作为评判准绳，选用表现最优的权重文件进行下游推理部署。

在深度学习研究与工程落地中，并行探索多组不同超参数配置的模型并在终局按验证集评测指标选优，属于标准通用范式。此外，划分给验证集的数据块总量亦需保持一定规模，避免因验证集过小导致评估波动过大。

## 历史溯源与致谢 (Acknowledgements)

本代码库最初源自牛津大学机器学习课程的 [Practical 6](https://github.com/oxford-cs-ml-2015/practical6)，而该作业又基于 Wojciech Zaremba 的 [learning to execute](https://github.com/wojciechz/learning_to_execute) 开源代码实现。此外，部分核心工程模块亦与其实验室同僚 [Justin Johnson](http://cs.stanford.edu/people/jcjohns/) 协作开发完成。

如需深入系统地学习 RNN 语言模型理论，推荐研读以下权威文献与讲座：

- Andrej Karpathy 关于 char-rnn 的[专题演讲视频](https://skillsmatter.com/skillscasts/6611-visualizing-and-understanding-recurrent-networks)
- Alex Graves 的经典论文 [《Generating Sequences With Recurrent Neural Networks》](http://arxiv.org/abs/1308.0850)
- Ilya Sutskever 的经典论文 [《Generating Text with Recurrent Neural Networks》](http://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf)
- Tomas Mikolov 的博士毕业论文 [《Statistical Language Models based on Neural Networks》](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)

## 许可证 (License)

[MIT 许可证](https://opensource.org/licenses/MIT)
