import json
import click
import logging
import code
from code import *


@click.command()
@click.option("--model_name", help="Name of the model")
@click.option("--num_epochs", default=10, help="Number of epochs")
@click.option("--hidden_dim", default=512, help="Dimension of each hidden layer")
@click.option("--num_layers", default=1, help="Number of hidden layers")
@click.option("--num_classes", default=3, help="Number of output classes")
@click.option(
    "--use_glove", default="False", help="Whether to initialise using glove embeddings"
)
@click.option("--batch_size", default=32, help="Batch size")
@click.option("--log_level", default="INFO", help="Log level (default: INFO)")
def main(
    model_name,
    num_epochs,
    hidden_dim,
    num_layers,
    num_classes,
    use_glove,
    batch_size,
    log_level,
):

    # Set logger config
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    (emb_dict, emb_dim), (
        dataloader_train,
        dataloader_valid,
        dataloader_test,
    ) = load_snli_dataset(batch_size)

    word_counts = build_word_counts(dataloader_train)
    index_map = build_index_map(word_counts)

    model = getattr(code, model_name)

    progress = run_snli(
        model,
        num_epochs=num_epochs,
        vocab_size=len(index_map),
        hidden_dim=hidden_dim,
        embedding_size=emb_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        use_glove=use_glove,
    )

    with open(
        f'data/{model_name}_{"glove" if use_glove == "True" else ""}.json', "w"
    ) as f:
        json.dump(progress, f)


if __name__ == "__main__":
    main()
