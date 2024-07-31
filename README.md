# c-af-form-experiment

This repo can be used to process forms into their constiuent questions. It is heavily 
inspired from the work here https://github.com/timpaul/form-extractor-prototype/tree/main

The notebook takes pdfs, converts to images and returns a list of questions from the 
form. It used Claude Sonnet 3.5. 


## Recreate the environment 

```sh
conda create -n forms python=3.12
conda activate forms
pip install -r requirements.txt
```

You will also need to install `poppler` for processing the pdfs. 
On a mac with homebrew 

```sh
brew install poppler
```

You will also need a `.env` file in the directory with the following, replacing the placeholder 
with a profile with Bedrock invoke permissions. 

```
AWS_DEFAULT_PROFILE=aws_user_with_bedrock
```