<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h2 align="center">Large Language Model Fine-tuning with Low-Rank Adaptation: A Performance Exploration</h3>
  <h1 align="center">The 16th ACM/SPEC International Conference on Performance Engineering (ICPE ’25), May 5–9, 2025, Toronto, Canada</h3>

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


## License

Distributed under the MIT License. See `LICENSE` for more information.

