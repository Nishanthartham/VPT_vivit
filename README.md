# VPT-Vivit Setup

1) Place the ckpt-10-class-SNR005 in the same VPT_vivit dir
    
## Environment settings

1) Use 'requirements.txt` to setup packages in python venv
  a) Run command python -m venv venv_vivit (inside VPT_vivit dir)
  b) source venv_vivit/bin/activate
  c) pip install -r requirements.txt

2) Go to file venv_vivit/lib/python3.10/site-packages/transformers/models/vivit/modeling_vivit.py and replace code from new_modeling_vivit.py to modeling_vivit.py from transformers package.

## Running VPT-Vivit

bash run.sh <log file name\>
