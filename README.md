# c-af-form-experiment

This repo can be used to process forms into their constiuent questions. It is heavily 
inspired from the work here https://github.com/timpaul/form-extractor-prototype/tree/main

The notebook takes pdfs, converts to images and returns a list of questions from the 
form. It uses Claude Sonnet 3.5. There is quite a lot of code to allow the calls to be
asynchronous and run in batches. The code saves its progress and the output to 
`batch_results.pickle` after each batch. 

If you get a failure, you can solve that - maybe by deleting the pdf - and rerun the cell
and it should pick up where it left off. If you want a fresh start delete the files and 
probabably the log. For a brand new run, run the below. 

```sh
rm batch_results.pkl 
rm form_processing.log
touch form_processing.log
```

You can check the logs while its running: 

```sh
tail -f form_processing.log 
```

## IAM

I had a lot of issues with our default way of accessing aws so I needed to make a new
user specific for this job. To replicate the role requires BedrockAll. Possibly this is 
too permisive as we are only invoking calls to Claude. 


## Prerequisites 

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

You will also need a `.env` file. I have created a special user for this purpose. 
Unimagininatively it is called bedrock-david-temp. Login to the console and retrieve
the key pair. You will need to run in `us-east-1` as it uses Sonnet 3.5
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-1

```
AWS_DEFAULT_PROFILE=aws_user_with_bedrock
```

## Notes on use

I have directly used the tool from timpauls repo. Its possible this could be adapted to 
include more types of field, eg. driving licence number. We probably dont need those 
text hints in the output either this would save us some output tokens. 

For a sample of 100 forms the token count was:

```
input_tokens     459788
output_tokens    106684
```

Which with Claude 3.5 comes to just under $3 or 0.30cents perfrom. For all the pds in 
the directory, estimated cost is $180. With this in mind its probably worth deciding 
if the current extraction schema is appropriate.