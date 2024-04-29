LinkBERT: A Knowledgeable Language Model Pretrained with Document Links

Overview

LinkBERT is a new pretrained language model (improvement of BERT) that captures document links such as hyperlinks and citation links to include knowledge that spans across multiple documents. Specifically, it was pretrained by feeding linked documents into the same language model context, besides using a single document as in BERT.

LinkBERT can be used as a drop-in replacement for BERT. It achieves better performance for general language understanding tasks (e.g. text classification), and is also particularly effective for knowledge-intensive tasks (e.g. question answering) and cross-document tasks (e.g. reading comprehension, document retrieval).







# shell comands to download raw data


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
