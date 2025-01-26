import time
import os
import random
import sys
import yaml
import torch

from transformers import EncoderDecoderModel, BertTokenizerFast, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import datasets


def main():

    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/sum_transformer.yaml'

    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    # 랜덤 시드 세팅
    if 'random_seed' in params:
        seed = params['random_seed']
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    tokenizer = BertTokenizerFast.from_pretrained(params['encoder_model']) # PLM 중 tokenizer 선택
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token

    # 데이터 로드
    # 데이터 로드
    if params['task'] == "C4":  # C4 데이터셋 로드
        train_data = datasets.load_dataset("c4", "en", split="train")
        val_data = datasets.load_dataset("c4", "en", split="validation[:10%]")

    elif params['task'] == "The_Pile":  # The Pile 데이터셋 로드
        train_data = datasets.load_dataset("the_pile", split="train")
        val_data = datasets.load_dataset("the_pile", split="validation[:10%]")

    elif params['task'] == "WikiText-103":  # WikiText-103 데이터셋 로드
        train_data = datasets.load_dataset("wikitext", "103", split="train")
        val_data = datasets.load_dataset("wikitext", "103", split="validation[:10%]")

    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels (highlights)
        inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=params['encoder_max_length'])
        outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=params['decoder_max_length'])

        batch["input_ids"] = inputs.input_ids # article 문서를 tokenize 한 단어 index들을 batch의 input_ids로 
        batch["attention_mask"] = inputs.attention_mask # encoder의 attention mask 할당
        batch["decoder_input_ids"] = outputs.input_ids # decoder에는 highlight 문장을 tokenizer 한 단어 index들을 
        batch["decoder_attention_mask"] = outputs.attention_mask # decoder의 attention mask 할당
        batch["labels"] = outputs.input_ids.copy()

        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
        # We have to make sure that the PAD token is ignored
        # batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in
        #                    batch["labels"]] # PAD를 무시하도록 -100 할당
        return batch

    if params['task'] == "C4":
        # C4 데이터셋 처리
        train_data = train_data.map(
            process_data_to_model_inputs,
            batched=True,
            batch_size=params['batch_size'],
            remove_columns=["text"]  # C4는 'text' 컬럼이 주요 텍스트 데이터임
        )
        train_data.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],  # 필요한 컬럼만 사용
        )

        val_data = val_data.map(
            process_data_to_model_inputs,
            batched=True,
            batch_size=params['batch_size'],
            remove_columns=["text"]
        )
        val_data.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

    elif params['task'] == "The_Pile":
        # The Pile 데이터셋 처리
        train_data = train_data.map(
            process_data_to_model_inputs,
            batched=True,
            batch_size=params['batch_size'],
            remove_columns=["text"]  # The Pile도 'text' 컬럼에 텍스트 데이터가 있음
        )
        train_data.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

        val_data = val_data.map(
            process_data_to_model_inputs,
            batched=True,
            batch_size=params['batch_size'],
            remove_columns=["text"]
        )
        val_data.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

    elif params['task'] == "WikiText-103":
        # WikiText-103 데이터셋 처리
        train_data = train_data.map(
            process_data_to_model_inputs,
            batched=True,
            batch_size=params['batch_size'],
            remove_columns=["text"]  # WikiText-103도 'text' 컬럼 사용
        )
        train_data.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

        val_data = val_data.map(
            process_data_to_model_inputs,
            batched=True,
            batch_size=params['batch_size'],
            remove_columns=["text"]
        )
        val_data.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

    model = EncoderDecoderModel.from_encoder_decoder_pretrained(params['encoder_model'], params['decoder_model'])
    # set special tokens
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # sensible parameters for beam search
    model.config.vocab_size = model.config.encoder.vocab_size
    model.config.max_length = 142
    model.config.min_length = 56
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.length_penalty = 2.0
    model.config.num_beams = params['num_beams']

    print(model)

    rouge = datasets.load_metric("rouge")
    def compute_metrics(pred):

        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # all unnecessary tokens are removed
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }

    # set training arguments - these params are not really tuned, feel free to change
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp)))

    # https://huggingface.co/transformers/v4.2.2/main_classes/trainer.html#transformers.TrainingArguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        evaluation_strategy="steps",  # 설정된 step 수 마다 평가
        per_device_train_batch_size=params['batch_size'],  # device 당 train batch size
        per_device_eval_batch_size=params['batch_size'],  # device 당 eval batch size
        learning_rate=params['optimizer_params'][params['optimizer']]['lr'],
        num_train_epochs=params['max_epochs'],
        logging_dir=out_dir,
        predict_with_generate=True,
        logging_steps=params['logging_steps'],  # logging 주기
        save_steps=params['save_steps'],  # save 주기
        eval_steps=params['eval_steps'],  # evaluation 주기
        warmup_steps=params['warmup_steps'],  # warmup 완료 step
        overwrite_output_dir=True,  # checkpoint에 덮어씌워 training을 계속 할지 여부
        save_total_limit=3,  # checkpoint 최대 저장 개수
        fp16=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,
    )
    trainer.train()


if __name__ == '__main__':
    main()