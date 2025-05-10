# Installation Instruction

Start by cloning the repo:
```bash
git clone https://github.com/evelynturri/pv-ai-detector.git
cd pv-ai-detector
```

## Virtual Environment
Build your environment to install all the requirements. You can use [venv](https://docs.python.org/3/library/venv.html) or [anaconda](https://www.anaconda.com/). 
The python version has been used in the project is the 3.10.8. 
We tested the code on linux with CUDA 12.1 and PyTorch 2.3.1, but it should work file also on different configurations.

### venv

```bash
python -m venv ai-pv
source ai-pv/bin/activate
```
### anaconda

You can create an anaconda environment called `repo` as below.
```bash
conda create -n ai-pv python=3.10.8
conda activate ai-pv
```

## Installation
After creating the virtual environment you can install all the required libraries. 

First install pytorch based on the CUDA used on your system. If you want to follow what we used, run this command, otherwise choose one on PyTorch [website](https://pytorch.org/get-started/previous-versions/):
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```
Then install all the other required packages by running:

```bash
pip install -r requirements.txt
```

## OpenAI API Key
Create a `.env` file in the project root with your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key
```
> Take a look at the [.env.example](.env.example) 