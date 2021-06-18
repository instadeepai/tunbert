#!/usr/bin/env bash

OUTPUT_SA_TDID_FOLDER_NAME="./finetuning_tsac"
DATA_FOLDER_NAME="./dev-data/sentiment_analysis_tsac"
BERT_FOLDER_NAME="./tf_tunbert"

read -p "Do you want to finetune and evaluate your language model for SA OR tunisian
dialect identifiaction task (y/n) ?" -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
	python models/bert-google/bert/run_classifier.py \
		--task_name=sst2 \
		--vocab_file=$BERT_FOLDER_NAME/vocab.txt \
		--bert_config_file=$BERT_FOLDER_NAME/bert_config.json \
		--init_checkpoint=$BERT_FOLDER_NAME/bert_model_step_616000.ckpt-154.data-00000-of-00001 \
		--do_train=True \
		--do_eval=True \
		--do_predict=False \
		--data_dir=$DATA_FOLDER_NAME/ \
		--train_batch_size=32 \
		--learning_rate=2e-5 \
		--num_train_epochs=5 \
		--output_dir=$OUTPUT_SA_TDID_FOLDER_NAME
fi

read -p "Do you want to make predictions (y/n)?" -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
	python models/bert-google/bert/run_classifier.py \
		--task_name=sst2 \
		--vocab_file=$BERT_FOLDER_NAME/vocab.txt \
		--bert_config_file=$BERT_FOLDER_NAME/bert_config.json \
		--init_checkpoint=$BERT_FOLDER_NAME/bert_model_step_616000.ckpt-154.data-00000-of-00001 \
		--do_train=False \
		--do_eval=False \
		--do_predict=True \
		--data_dir=$DATA_FOLDER_NAME/ \
		--train_batch_size=32 \
		--learning_rate=2e-5 \
		--num_train_epochs=5 \
		--output_dir=$OUTPUT_SA_TDID_FOLDER_NAME
fi
