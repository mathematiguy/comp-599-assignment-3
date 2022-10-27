import json
import click
import logging

import code
from code import *
import pandas as pd
import seaborn as sns

# Global hyperparams
@click.command()
@click.option("--model_name", help="Name of the model")
@click.option("--text_file", help="Text file for training")
@click.option("--num_epochs", default=100, help="Number of epochs")
@click.option("--batch_size", default=512, help="Batch size")
@click.option("--out_seq_len", default=200, help="Generated sequence length")
@click.option("--hidden_size", default=512, help="Hidden layer width")
@click.option("--embedding_size", default=300, help="Word embedding size")
@click.option("--lr", default=0.002, type=float, help="Batch size")
@click.option("--seq_len", default=100, help="Batch size")
@click.option("--log_level", default="INFO", help="Log level (default: INFO)")
def main(
    model_name,
    text_file,
    num_epochs,
    batch_size,
    out_seq_len,
    hidden_size,
    embedding_size,
    lr,
    seq_len,
    log_level,
):

    # Set logger config
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # code to initialize dataloader, model
    dataset = CharSeqDataloader(
        filepath=text_file, seq_len=seq_len, examples_per_epoch=batch_size
    )
    model = CharRNN(
        n_chars=len(dataset.unique_chars),
        embedding_size=embedding_size,
        hidden_size=hidden_size,
    )
    model.to(device)

    # Train the model
    progress = train(
        model,
        dataset,
        lr=lr,
        num_epochs=num_epochs,
        out_seq_len=out_seq_len,
        sample_file=f"data/{model_name}.txt",
    )

    with open(f"data/{model_name}.json", "w") as f:
        json.dump(progress, f)


if __name__ == "__main__":
    main()
