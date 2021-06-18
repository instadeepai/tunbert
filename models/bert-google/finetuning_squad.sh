#!/usr/bin/env bash


OUTPUT_SQUAD_FOLDER_NAME="./finetuning_trcd"
DATA_FOLDER_NAME="./dev-data/question_answering_trcd"
BERT_FOLDER_NAME="./tf_tunbert"

mkdir -p $OUTPUT_SQUAD_FOLDER_NAME


read -p "Do you want to finetune and make predictions AraBERT for Q&A (yes/no)?" -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
	python models/bert-google/bert/run_squad.py \
	  --vocab_file=$BERT_FOLDER_NAME/vocab.txt \
	  --bert_config_file=$BERT_FOLDER_NAME/bert_config.json \
	  --init_checkpoint=$BERT_FOLDER_NAME/bert_model_step_616000.ckpt-154.data-00000-of-00001 \
	  --do_train=True \
	  --train_file=$DATA_FOLDER_NAME/train.json \
	  --do_predict=True \
	  --predict_file=$DATA_FOLDER_NAME/test.json \
	  --train_batch_size=2 \
	  --learning_rate=3e-5 \
	  --num_train_epochs=2 \
	  --max_seq_length=384 \
	  --doc_stride=128 \
	  --do_lower_case=False\
	  --output_dir=$OUTPUT_SQUAD_FOLDER_NAME/
fi

read -p "Do you want to evaluate Q&A (yes/no)?" -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
	python models/bert-google/bert/evaluate.py \
	  --dataset_file=$DATA_FOLDER_NAME/test.json \
	  --prediction_file=$OUTPUT_SQUAD_FOLDER_NAME/predictions.json
fi
