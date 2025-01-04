# moe-interp

## data

### math, physics, biology
these were scraped from the arxiv.org until we reached 200k tokens on the most recent papers of each category.

## github
`data/github.txt` was retrieved from OLMoE's repository. 

### legal
`data/legal.txt` was retrieved from [CUAD_v1 dataset](https://www.atticusprojectai.org/cuad). it includes first 15 documents from the dataset (~200k tokens)

## plots
`plots/` contains the plots for the expert distribution for all experiments. (% of tokens going to a particular expert)

## results
`results/` contains the model outputs for various experiments like original model generation and swapped experts model generation.

## experiments

`expert-routing.ipynb` contains the code for the expert routing and expert distribution experiments where we tracked the % of tokens that went to a particular expert. 

`swap-experts.ipynb` contains the code for the expert swapping experiments where we swapped experts between the same and across different layers and compared the model outputs on various inputs.

`swap-routing.ipynb` contains the code for the expert distribution after swapping experts in various combinations.

`swap-mlp-transformer.ipynb` contains the code for the model outputs after swapping MLPs of various layers in dense transformers.

`zero-out-exp.ipynb` contains the code for the model outputs after zeroing out the experts in the model in various combinations.

`zero-out-mlp-transformer.ipynb` contains the code for the model outputs after zeroing out the MLPs of various layers in dense transformers.