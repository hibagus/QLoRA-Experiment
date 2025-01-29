<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h1 align="center">Large Language Model Fine-tuning with Low-Rank Adaptation: A Performance Exploration</h3>
  <h3 align="center">The 16th ACM/SPEC International Conference on Performance Engineering (ICPE ’25), May 5–9, 2025, Toronto, Canada</h3>

  <p align="center">
    Bagus Hanindhito, The University of Texas at Austin, Austin, Texas, USA
    <br />
    Bhavesh Patel, Dell Technologies, Round Rock, Texas, USA
    <br />
    Lizy K. John, The University of Texas at Austin, Austin, Texas, USA
    <br />
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-this-repository">About This Repository</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#known-problem-and-solution">Known Problem and Solution</a></li>
      <ul>
        <li><a href="#bitsandbytes-cuda-runtime-error">BitsandBytes CUDA Runtime Error</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    <li><a href="#license">License</a></li>
  </ol>
</details>


## About This Repository
This repository contains scripts, log files, and guidance on how to reproduce the experiments done in our paper.
We followed the ICPE Artifact Evaluation Track to improve reproducibility of our papers.
However, as the contents provided here are tested with specific version of libraries, they may not work with newer version of libraries. 

### Built With
This section should list any major frameworks that you built your project using. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [HuggingFace Transformers](https://github.com/huggingface/transformers)
* [HuggingFace Evaluate](https://github.com/huggingface/evaluate)
* [HuggingFace PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
* [HuggingFace Accelerate](https://github.com/huggingface/accelerate)
* [Microsoft DeepSpeed](https://github.com/microsoft/DeepSpeed)
* [BitsandBytes](https://github.com/bitsandbytes-foundation/bitsandbytes)

## Getting Started
Below are the step-by-step to reproduce the result in our experiments.

### Prerequisites
We assume a compute environment with the following configurations:
* Operating system: Ubuntu 22.04
* NVIDIA driver: 535.86.10
* CUDA Toolkit Version: 12.2
* Anaconda 2023.07.2

Different configurations with newer version of driver, CUDA Toolkit, and Anaconda may also work.

You can also use different Python package manager / virtual environment, but for simplicity, we use Anaconda for the following steps.

### Installation

1. Clone the repository to your local computer.
   ```sh
   git clone https://github.com/hibagus/QLoRA-Experiment.git
   cd QLoRA-Experiment
   ```
2. Create new virtual environment on Anaconda using the provided `environment.yml`.
   Steps below assume that your new virtual environment is named "qlora".
   ```sh
   conda env create -f environment.yml -n qlora
   conda activate qlora
   ```

## Usage

Run the fine-tuning scripts provided within the repository.

The scripts are organized into three folders: `std`, `lora`, and `qlora`.

Step below run QLoRA fine-tuning on Llama2 model with 7 billion parameters.
```sh
cd scripts/qlora
./finetune_llama2_7b_qlora_fp4_bf16.sh
```

The script will generate log file inside `log` folder and TensorBoard log file inside `output` folder.

We also provide some examples of log files accessible below.
```sh
cd logfiles
```

## Known Problem and Solution

Below are several known problems and solutions.

### BitsandBytes CUDA Runtime Error
* BitsandBytes complain CUDA SETUP error with the following message:
```sh
================================================ERROR=====================================
CUDA SETUP: CUDA detection failed! Possible reasons:
1. CUDA driver not installed
2. CUDA not installed
3. You have multiple conflicting CUDA libraries
4. Required library not pre-compiled for this bitsandbytes release!
CUDA SETUP: If you compiled from source, try again with `make CUDA_VERSION=DETECTED_CUDA_VERSION` for example, `make CUDA_VERSION=113`.
CUDA SETUP: The CUDA version for the compile might depend on your conda install. Inspect CUDA version via `conda list | grep cuda`.
================================================================================
```
  This happens when you have newer version of CUDA Toolkit installed (i.e., newer than version 12.2).

* There are two solutions described below.
  1. Build BitsandBytes from source. This is the recommended way, but the most difficult to accomplish. It may not be straightforward since your PyTorch must match the version of CUDA Toolkit you've installed.
  ```sh
  git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
  cd bitsandbytes
  git checkout 0.40.0 # You can always try newer version if you wish
  CUDA_VERSION=XXX # Replace XXX with your CUDA Toolkit version
  python setup.py install
  ```
  2. Force BitsandBytes to use shared object library (`.so`) precompiled for older CUDA version. This is the easiest and would work for CUDA Toolkit with different minor version (i.e., 12.2 with 12.4). However, it may not reflect the performance improvements that might be provided with newer CUDA Toolkit version.
  ```sh
  # assuming you use Anaconda and you installed anaconda in your home directory and you use CUDA Toolkit version 12.8
  cd /home/[username]/anaconda3/envs/[virtual_env_name]/lib/python3.11/site-packages/bitsandbytes
  # replace [username] with your username
  # replace [virtual_env_name] with the name you've given.
  ln -s libbitsandbytes_cuda122.so libbitsandbytes_cuda128.so
  ```


## License

Distributed under the MIT License. See `LICENSE` for more information.

