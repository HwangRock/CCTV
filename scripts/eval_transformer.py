import torch
import yaml
import sys
import os
import datasets
from transformers import BertTokenizer, EncoderDecoderModel

def main():
    print('RNN for machine translation evaluation')

    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/sum_transformer.yaml'

    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    timestamp = "1696999441"
    checkpoint = 'checkpoint-26500'
    out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp, checkpoint)))

    tokenizer = BertTokenizer.from_pretrained(params['encoder_model'])
    model = EncoderDecoderModel.from_pretrained(out_dir)
    model.to(device)

    test_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="test")

    batch_size = params['batch_size']

    # map data correctly
    def generate_summary(batch):
        # Tokenizer will automatically set [BOS] <text> [EOS]
        # cut off at BERT max length 512
        inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")

        outputs = model.generate(input_ids, attention_mask=attention_mask)

        # all special tokens including will be removed
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        batch["pred"] = output_str

        return batch

    results = test_data.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["article"])

    pred_str = results["pred"]
    label_str = results["highlights"]

    rouge = datasets.load_metric("rouge")
    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1", "rouge2", "rougeL"])

    # Print all ROUGE scores
    print("ROUGE-1:", rouge_output["rouge1"].mid)
    print("ROUGE-2:", rouge_output["rouge2"].mid)
    print("ROUGE-L:", rouge_output["rougeL"].mid)


if __name__ == "__main__":
    main()
