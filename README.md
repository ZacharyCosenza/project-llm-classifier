# run_1_distilBERT.ipynb

For this run I trained a small neural network head on top of a DistilBERT language model with hidden size 768, 6 layers, and 12 attention heads. The model is very simple: text is tokenized using `distilbert-base-uncased` with max length 512, passed to the language model where I take the last layer output (size 768) for response A and B, concatenate them, then pass them into the head. This produces logits which are classified as A wins/B wins/Tie. For batch size 32, 3 epochs with AdamW and learning rate of 1e-5 resulted in accuracy no better than chance (~33%). After this initial experiment, the next logical step is to inject information about the prompt into the training. The most challenging aspect was getting Intel's XPUs to work correctly on a Windows 11 machine.

# run_2_appendprompt.ipynb

For this run I used the previous run 1 architecture but appended the tokenized prompt to the beginning of the responses. This increased the compute requirements, which required me to reduce the batch size to 2 and run for a single epoch to conserve memory on my machine and reduce experiment time. The accuracy of the test set was ~45%, which was a considerable improvement over run 1. However, it is unclear if that is due to changing batch size or the prompt appending. Now that we have a direction, the next step is to improve the compute situation so more and larger experiments can be run.

# Attempt to Increase Compute

Seeing as I deleted my AWS account and cannot access it, I turned to runpod.io for access to GPUs. I also switched to WSL for development on my Windows 11 machine, and plan to run tests locally and deploy training runs on the GPUs. Another difficulty is setting up a rational workflow between local and deployment environments. I am able to SSH into the runpod machines, but cannot easily connect to VSCode's workspace, so I must switch to production runs using `.py` files rather than `.ipynb`. Setting up a virtual environment on runpod was also a pain. I ended up putting together a `setup.sh` file to standardize everything. Thankfully it worked on both CPU and GPU machines. The biggest problem was getting SSH to work correctly. I think VSCode and WSL are having trouble playing nice because the config is auto-generated in a Windows path (`user/.ssh`) and the ID files are in a Linux path (`~/.ssh`), so the config cannot find the ID files. I fixed this by adding the ID files to the Windows `.ssh` folder. More rigorous study of what exactly VSCode is using for SSH operations is needed. This ended up not mattering because I ended up using GitHub and Kaggle to transfer files. After two days I got CPU and GPU (1x A40) to work with ~$4/month in storage. 

# Return to run_1_distilBERT.ipynb

Because I was using just the terminal to run experiments in the CPU/GPU machines on runpod, I needed to convert the notebook to a file and introduce more sophisticated saving and quality-of-life features (printing, logging, resuming from checkpoints, etc.). I also created `startup.sh` to ensure consistency across my runpod CPU/GPU setup (installing dependencies, setting up the environment, etc.). AI was very helpful in quickly iterating and adding these features. I suspect that my experimental conditions changed when going from notebook to Python file because accuracy shot up to 50%. My two hypotheses are (1) the `max_length` of the tokenizer changed and/or (2) the train/test/validation split may have changed. I lowered `max_length` to 128 and raised test and validation split to 10% each (note: this only reduced accuracy to 46%). The implementation was pretty rough to create a baseline, so let's move onto run_2.

# Return to run_2_appendprompt.ipynb

This experiment just involved concatenating the prompt and response message and passing through the tokenization. This adds context at the expense of exceeding the `max_length` limit of the tokenizer/model combination. On my old local XPU this caused memory issues, so I ran with batch size of 2 and only managed to get through a single epoch. Moving to the more powerful A40, I am able to use standard batch size 32 for 3 epochs without issue. Training finished in less than an hour with test accuracy of 47.3%. 

# Beyond the Baseline

Appending the prompt to the response is one way of improving performance by more efficiently using the given model's context. This will act as a good baseline. Right now I'm using `distilbert-base-uncased` from Hugging Face. According to [the DistilBERT paper](https://arxiv.org/abs/1910.01108), DistilBERT has 66M parameters. Let's run experiments along the following lines: (1) number of tuned parameters, (2) number of total parameters, (3) pre-training.

## Experiment Set 1: Parameter Tuning with DistilBERT-base-uncased
- Random initialization: 35.1%
- Baseline (full training): 47.3%
- Fully frozen base model: 45.3%
- Bottom layers frozen: 46.9%

Clearly training on more parameters has a positive effect on accuracy. As a next step we are trying out models with greater raw capacity such as BERT and BERT-large. I also have started to organize the codebase now that some experiments are more stable (moving dataloader and model classes to `core.py`). Running these experiments took far more time than a 1x A40 could handle, so I upgraded to 4x A40 which required some infrastructure changes and use of tmux to run sessions while I was away. For future reference, I needed to make the following changes to the code for DDP (multi-GPU for PyTorch):

1. Run with `torchrun --nproc_per_node=4 python_script.py...`
2. Get local ranks and set communication parameters:
   ```python
   DEVICE = int(os.environ["LOCAL_RANK"])
   torch.cuda.set_device(DEVICE)
   dist.init_process_group(backend="gloo")
   ```
3. Modify batch size: `BATCH_SIZE = max(1, args.batch // WORLD_SIZE)`
4. Only save and log if `RANK == 0` (main process)
5. `if USE_DDP: dist.barrier()` was needed to make sure `RANK == 0` goes first
6. Wrap dataloaders in `DistributedSampler` and model in `DDP(model, device_ids=[DEVICE], find_unused_parameters=True)`. This turned out to be critical as my language model implementation does not use some parameters during inference.
7. `dist.broadcast(start_epoch_tensor, src=0)` and `train_sampler.set_epoch(epoch)` needed during training
8. `dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)` needed for non-training calculations

## Experiment Set 2: Model Size with BERT Family
Using BERT models to study the effect of model size, we found some improvement by going with larger models:
- DistilBERT-base: 47.3%
- BERT-base: 48.35%
- BERT-large: 48.43%

## Experiment Set 3: Different Pre-training Approaches
We tested similar-sized models with different pre-training strategies:

**RoBERTa-base**: Uses dynamic masking with 15% MLM (80% replaced with `<mask>`, 10% replaced by random token, 10% unchanged). Case-sensitive English. Training data includes BookCorpus, English Wikipedia, CC-News, OpenWebText, and Stories (~160GB total). Tokenized using BPE with new documents marked with `<s>`. **Accuracy: 48.6%**

**ELECTRA-base-discriminator** (`google/electra-base-discriminator`): Uses a generator-discriminator architecture where a generator language model creates errors in the data, and the discriminator is trained to detect real vs. fake tokens (replaced-token detection rather than masked language modeling). **Accuracy: 49.3%**

**DeBERTa-base** (`microsoft/deberta-base`): Introduces disentangled attention mechanism where each word is represented using two vectors encoding content and position separately. Uses an enhanced mask decoder incorporating absolute positions. Trained on 80GB of data with similar architecture to BERT (768 hidden size, 12 layers, 12 attention heads). The disentangled attention allows the model to better capture relative positions between tokens. **Accuracy: 35.1%**

**ALBERT-base-v2** (`albert-base-v2`): A "Lite BERT" using factorized embedding parameterization (splits vocabulary embedding matrix into two smaller matrices: V×E and E×H instead of V×H, where E=128) and cross-layer parameter sharing (all 12 layers share the same weights). This reduces parameters from BERT's 110M to only 11M while maintaining performance. Trained on English Wikipedia and BookCorpus. Version 2 includes improved dropout rates, additional training data, and longer training. **Accuracy: 49.1%**

# PyTorch Lightning

I figured I'd implement the current method in PyTorch Lightning (partially to test out Claude Code). The new way to run is something like:
```bash
python run_6_light.py --epochs 3 --batch 32 --devices 2 --strategy ddp
```

It seems the overall value here is it simplified the syntax around using multiple GPUs (no more `torchrun` and fiddling with DDP). 