import argparse
import os
import sys

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
import pandas as pd

def add_base_args(parser):
    parser.add_argument("--model", help="Model name to use for embedding.",
                        type=str, required=True)
    parser.add_argument("--alias", help="Alias to use for fine-tuned model.",
                        type=str, required=True)
    parser.add_argument("--dataset", help="Name of fine-tuning dataset in datasets/finetuning.",
                        type=str, required=True)
    parser.add_argument("--epochs", help="Number of epochs to fine-tune for.",
                        type=int, default=3)
    parser.add_argument("--save_strategy", help="Save strategy for checkpoints.",
                        type=str, choices=["no", "epoch"], default="no")
    return parser


def main():
    parser = argparse.ArgumentParser() # Can be simplified
    parser = add_base_args(parser)

    args = parser.parse_args()

    model_name = args.model
    dataset = args.dataset
    finetuned_alias = args.alias
    save_strategy=args.save_strategy

    epochs = args.epochs

    print(f"Fine-tuning model: {model_name}")

    # 1. Load model to finetune
    model = SentenceTransformer(model_name)

    # 2. Load finetuning dataset
    file_path = os.path.join("datasets/finetuning", f"{dataset}.xlsx")
    df = pd.read_excel(file_path)
    train_dataset = Dataset.from_pandas(df)

    # 3. Loss function
    loss = MultipleNegativesRankingLoss(model) #Create getLoss in utils

    # 4. Training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"models/{finetuned_alias}",
        # Optional training parameters:
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if GPU can't handle FP16
        bf16=False,  # Set to True if GPU supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicates
        save_strategy=save_strategy
    )

    # 5. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()

    # 6. Save the trained model
    model.save_pretrained(f"models/{finetuned_alias}_{epochs}")


if __name__ == "__main__":
    main()