# LinkBERT: Fine Tuning Pre-trained Language Model with Document Links

# Overview

LinkBERT is a new pretrained language model (improvement of BERT) that captures document links such as hyperlinks and citation links to include knowledge that spans across multiple documents. Specifically, it was pretrained by feeding linked documents into the same language model context, besides using a single document as in BERT.


# Shell comands to download raw data


mkdir -p raw_data

OUTPUT=raw_data/mrqa/train  
mkdir -p $OUTPUT  
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/HotpotQA.jsonl.gz -O $OUTPUT/HotpotQA.jsonl.gz  
gzip -d $OUTPUT/HotpotQA.jsonl.gz  
OUTPUT=raw_data/mrqa/dev  
mkdir -p $OUTPUT  
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/HotpotQA.jsonl.gz -O $OUTPUT/HotpotQA.jsonl.gz  
gzip -d $OUTPUT/HotpotQA.jsonl.gz


### After this, Run [preprocessNLPHotpotQA.py](preprocessNLPHotpotQA.py) to preprocess the raw_data.

# Set up environment and data

Environment

Run the following commands to create a conda environment:

conda create -n linkbert python=3.8
source activate linkbert
pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install transformers==4.9.1 datasets==1.11.0 fairscale==0.4.0 wandb sklearn seqeval

### Finally, run the following commands to run the model

export MODEL=LinkBERT-base  
export MODEL_PATH=michiyasunaga/$MODEL  
task=hotpot_hf  
datadir=../data/qa/$task  
outdir=runs/$task/$MODEL  
mkdir -p $outdir  
python3 -u qa/RunNLPfinal.py --model_name_or_path $MODEL_PATH \  
    --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \  
    --do_train --do_eval --do_predict --preprocessing_num_workers 10 \  
    --per_device_train_batch_size 12 --gradient_accumulation_steps 2 \  
    --learning_rate 3e-5 --num_train_epochs 4 --max_seq_length 384 --doc_stride 128 \  
    --save_strategy no --evaluation_strategy steps --eval_steps 1000 --output_dir $outdir --overwrite_output_dir \  
  |& tee $outdir/log.txt &
