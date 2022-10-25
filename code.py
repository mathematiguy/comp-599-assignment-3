import pdb
import re
import random
import numpy as np
import pandas as pd

from itertools import islice
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical

from datasets import load_dataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##### PROVIDED CODE #####


def tokenize(
    text: "list[str]", max_length: int = None, normalize: bool = True
) -> "list[list[str]]":
    import re

    if normalize:
        regexp = re.compile("[^a-zA-Z ]+")
        # Lowercase, Remove non-alphanum
        text = [regexp.sub("", t.lower()) for t in text]
    return [t.split()[:max_length] for t in text]


def build_index_map(
    word_counts: "dict[str, int]", max_words: int = None
) -> "dict[str, int]":
    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    if max_words:
        sorted_counts = sorted_counts[: max_words - 1]
    sorted_words = ["[PAD]"] + [item[0] for item in sorted_counts]
    return {word: ix for ix, word in enumerate(sorted_words)}


def build_word_counts(dataloader) -> "dict[str, int]":
    word_counts = {}
    for batch in dataloader:
        for words in tokenize(batch["premise"] + batch["hypothesis"]):
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts


def tokens_to_ix(
    tokens: "list[list[str]]", index_map: "dict[str, int]"
) -> "list[list[int]]":
    return [
        [index_map[word] for word in words if word in index_map] for words in tokens
    ]


##### END PROVIDED CODE #####


class CharSeqDataloader:
    def __init__(self, filepath, seq_len, examples_per_epoch):
        self.text = self.load_text(filepath)
        self.num_chars = len(self.text)
        self.unique_chars = self.get_unique_chars(self.text)
        self.vocab_size = len(self.unique_chars)
        self.mappings = self.generate_char_mappings(self.unique_chars)
        self.seq_len = seq_len
        self.examples_per_epoch = examples_per_epoch

    def load_text(self, filepath):
        with open(filepath, "r") as f:
            return f.read()

    def get_unique_chars(self, text):
        unique_chars = set(ch for ch in text)
        return sorted(unique_chars)

    def generate_char_mappings(self, uq):
        return {
            "idx_to_char": dict(zip(range(len(uq)), uq)),
            "char_to_idx": dict(zip(uq, range(len(uq)))),
        }

    def convert_seq_to_indices(self, seq):
        return [self.mappings["char_to_idx"][ch] for ch in seq]

    def convert_indices_to_seq(self, seq):
        return [self.mappings["idx_to_char"][ix] for ix in seq]

    def get_example(self):
        # Choose a start point at random
        start_idx = random.choice(range(self.num_chars - self.seq_len - 1))

        # Extract text and convert to list of chars
        in_string = list(self.text[start_idx : start_idx + self.seq_len])
        target_string = list(self.text[start_idx + 1 : start_idx + self.seq_len + 1])

        # Convert chars to idx sequences
        in_seq = self.convert_seq_to_indices(in_string)
        target_seq = self.convert_seq_to_indices(target_string)

        # Return as int tensors
        yield torch.tensor(in_seq, dtype=torch.int32), torch.tensor(
            target_seq, dtype=torch.int32
        )


class CharRNN(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharRNN, self).__init__()

        # Set hyperparameters
        self.hidden_size = hidden_size
        self.n_chars = n_chars
        self.embedding_size = embedding_size

        # Initialise layers
        self.hidden = torch.zeros(self.hidden_size)
        self.embedding_layer = nn.Embedding(self.n_chars, self.embedding_size)
        self.waa = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.wax = nn.Linear(self.embedding_size, self.hidden_size, bias=True)
        self.wya = nn.Linear(self.hidden_size, self.n_chars, bias=True)

    def rnn_cell(self, i, h):
        h_new = torch.tanh(self.waa(h) + self.wax(i))
        o = self.wya(h_new)
        return o, h_new

    def forward(self, input_seq, hidden=None):
        if hidden is None:
            hidden = self.hidden

        # Embed the input sequence
        input_seq = self.embedding_layer(input_seq)

        # Generate outputs + hidden states
        outputs = []
        hiddens = []
        for i in range(len(input_seq)):
            output, hidden = self.rnn_cell(input_seq[i, :], hidden)
            outputs.append(output)
            hiddens.append(hidden)

        # Stack the outputs
        out = torch.stack(outputs)
        hidden_last = hiddens[-1]

        return out, hidden_last

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)

    # Old rnn_cell based sample_sequence
    def sample_sequence(self, starting_char, seq_len, temp=0.5):
        generated_seq = torch.tensor([starting_char])
        embedded = self.embedding_layer(generated_seq)
        hidden = self.hidden

        # Generate outputs + hidden states
        for i in range(seq_len):
            output, hidden = self.rnn_cell(embedded[i, :], hidden)
            char_probs = F.softmax(output / temp, 0)
            next_char = Categorical(probs=char_probs).sample()
            generated_seq = torch.cat((generated_seq, next_char.flatten()))
            embedded = torch.cat(
                (embedded, self.embedding_layer(next_char).unsqueeze(0))
            )

        return generated_seq.tolist()


class CharLSTM(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_chars = n_chars

        self.hidden_input_size = self.hidden_size + self.embedding_size
        self.embedding_layer = nn.Embedding(self.n_chars, self.embedding_size)
        self.forget_gate = nn.Linear(
            self.hidden_input_size, self.hidden_size, bias=True
        )
        self.input_gate = nn.Linear(self.hidden_input_size, self.hidden_size, bias=True)
        self.output_gate = nn.Linear(
            self.hidden_input_size, self.hidden_size, bias=True
        )
        self.cell_state_layer = nn.Linear(
            self.hidden_input_size, self.hidden_size, bias=True
        )
        self.fc_output = nn.Linear(self.hidden_size, self.n_chars, bias=True)

    def forward(self, input_seq, hidden=None, cell=None):
        # Initialise hidden + cell state
        if hidden == None:
            hidden = torch.zeros(self.hidden_size)
        if cell == None:
            cell = torch.zeros(self.hidden_size)

        # Embed the input sequence
        input_seq = self.embedding_layer(input_seq)

        # Generate outputs + hidden + cell states
        outputs = []
        hiddens = []
        cells = []
        for i in range(len(input_seq)):
            output, hidden, cell = self.lstm_cell(input_seq[i, :], hidden, cell)
            outputs.append(output)
            hiddens.append(hidden)
            cells.append(cell)

        # Shape the outputs
        out_seq = torch.stack(outputs)
        hidden_last = hiddens[-1]
        cell_last = cells[-1]

        return out_seq, hidden_last, cell_last

    def lstm_cell(self, i, h, c):
        """
        Implement an lstm cell given input, hidden state + cell state
        """

        h_i = torch.cat((i, h))
        forget_gate = torch.sigmoid(self.forget_gate(h_i))
        input_gate = torch.sigmoid(self.input_gate(h_i))
        output_gate = torch.sigmoid(self.output_gate(h_i))

        cell_state = torch.tanh(self.cell_state_layer(h_i))
        c_new = forget_gate * c + input_gate * cell_state

        h_new = output_gate * torch.tanh(c_new)
        o = self.fc_output(h_new)

        return o, h_new, c_new

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def sample_sequence(self, starting_char, seq_len, temp=0.5):
        generated_seq = [starting_char]
        hidden = None
        cell = None
        for _ in range(seq_len):
            out, hidden, cell = self.forward(torch.tensor(generated_seq), hidden, cell)
            char_probs = F.softmax(out[-1] / temp, dim=0)
            generated_seq.append(Categorical(probs=char_probs).sample().tolist())
        return generated_seq


def train(model, dataset, lr, out_seq_len, num_epochs):

    # code to initialize optimizer, loss function
    optimizer = model.get_optimizer(lr=lr)
    loss_func = model.get_loss_function()

    starting_chars, starting_char_counts = zip(
        *Counter([ch for ch in dataset.text if re.match("[A-Z]", ch)]).items()
    )

    n = 0
    running_loss = 0
    for epoch in range(num_epochs):
        for in_seq, out_seq in dataset.get_example():
            # 1. Apply the RNN to the incoming sequence
            pred_seq, *_ = model.forward(in_seq)

            # 2. Use the loss function to calculate the loss on the model’s output
            current_loss = loss_func(pred_seq, out_seq.long())

            # 3. Zero the gradients of the optimizer
            optimizer.zero_grad()

            # 4. Perform a backward pass (calling .backward())
            current_loss.backward()

            # 5. Step the weights of the model via the optimizer (.step())
            optimizer.step()

            # 6. Add the current loss to the running loss
            running_loss += current_loss

            n += 1

        # print info every X examples
        print(f"Epoch {epoch}. Running loss so far: {(running_loss/n):.8f}")

        print("\n-------------SAMPLE FROM MODEL-------------")

        # code to sample a sequence from your model randomly
        with torch.no_grad():
            starting_char = random.choices(
                starting_chars, weights=starting_char_counts, k=1
            )
            starting_char = dataset.convert_seq_to_indices(starting_char)[0]
            generated_seq = model.sample_sequence(starting_char, out_seq_len, temp=0.9)
            print("".join(dataset.convert_indices_to_seq(generated_seq)))

        print("\n------------/SAMPLE FROM MODEL/------------")

        n = 0
        running_loss = 0

    return None  # return model optionally


def run_char_rnn():
    hidden_size = 512
    embedding_size = 300
    seq_len = 100
    lr = 0.002
    num_epochs = 100
    epoch_size = 10  # one epoch is this # of examples
    out_seq_len = 200
    data_path = "./data/shakespeare.txt"

    # code to initialize dataloader, model
    dataset = CharSeqDataloader(
        filepath=data_path, seq_len=seq_len, examples_per_epoch=epoch_size
    )
    model = CharRNN(
        n_chars=len(dataset.unique_chars),
        embedding_size=embedding_size,
        hidden_size=hidden_size,
    )

    # Train the model
    train(model, dataset, lr=lr, out_seq_len=out_seq_len, num_epochs=num_epochs)


def run_char_lstm():
    hidden_size = 512
    embedding_size = 300
    seq_len = 100
    lr = 0.002
    num_epochs = 100
    epoch_size = 10
    out_seq_len = 200
    data_path = "./data/shakespeare.txt"

    # code to initialize dataloader, model
    dataset = CharSeqDataloader(
        filepath=data_path, seq_len=seq_len, examples_per_epoch=epoch_size
    )
    model = CharLSTM(
        n_chars=len(dataset.unique_chars),
        embedding_size=embedding_size,
        hidden_size=hidden_size,
    )

    train(model, dataset, lr=lr, out_seq_len=out_seq_len, num_epochs=num_epochs)


def fix_padding(batch_premises, batch_hypotheses):

    batch_premises = [torch.tensor(premise, dtype=torch.int32) for premise in batch_premises]
    batch_hypotheses = [torch.tensor(hypothesis, dtype=torch.int32) for hypothesis in batch_hypotheses]

    batch_premises_reversed = [toks.flip(dims=(0,)) for toks in batch_premises]
    batch_hypotheses_reversed = [toks.flip(dims=(0,)) for toks in batch_hypotheses]

    batch_premises = pad_sequence(batch_premises, batch_first=True)
    batch_hypotheses = pad_sequence(batch_hypotheses, batch_first=True)

    batch_premises_reversed = pad_sequence(batch_premises_reversed, batch_first=True)
    batch_hypotheses_reversed = pad_sequence(batch_hypotheses_reversed, batch_first=True)

    return (
        batch_premises,
        batch_hypotheses,
        batch_premises_reversed,
        batch_hypotheses_reversed,
    )


def create_embedding_matrix(word_index, emb_dict, emb_dim):
    ix_to_word = {v: k for k, v in word_index.items()}

    # Convert numpy tensors to pytorch and initialize as zero if missing
    for word, ix in word_index.items():
        try:
            emb_dict[word] = emb_dict[word]
        except KeyError:
            emb_dict[word] = np.zeros(emb_dim)

    embeds = [torch.tensor(emb_dict[ix_to_word[i]], dtype=torch.float32) for i in range(len(word_index))]

    return torch.stack(embeds)


def evaluate(model, dataloader, index_map):

    num_datapoints = 0
    num_correct = 0
    with torch.no_grad():
        for batch in dataloader:
             premises, hypotheses, labels = batch.values()
             premises = tokenize(premises)
             hypotheses = tokenize(hypotheses)

             batch_premises = tokens_to_ix(premises, index_map)
             batch_hypotheses = tokens_to_ix(hypotheses, index_map)

             # 1. Apply the RNN to the incoming sequence
             try:
                 predictions = model.forward(batch_premises, batch_hypotheses)
             except RuntimeError:
                 pdb.set_trace()

             # 2. Count the correct labels
             predicted_labels = torch.argmax(predictions, axis=1)
             num_correct += torch.sum(predicted_labels == labels).tolist()
             num_datapoints += labels.shape[0]

    return num_correct / num_datapoints


class UniLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes):
        super(UniLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.int_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, num_classes)
        self.embedding_layer = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

    def forward(self, a, b):

        a, b, a_rev, b_rev = fix_padding(a, b)

        a_embed = self.embedding_layer(a)
        b_embed = self.embedding_layer(b)

        a_lstm_output, (a_lstm_h, a_lstm_c) = self.lstm(a_embed)
        b_lstm_output, (b_lstm_h, b_lstm_c) = self.lstm(b_embed)

        ab_cat = torch.cat((a_lstm_c, b_lstm_c), dim=2).squeeze(dim=0)
        ab_int = nn.ReLU()(self.int_layer(ab_cat))
        ab_out = self.out_layer(ab_int)

        return ab_out

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)


class ShallowBiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes):
        super(ShallowBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.lstm_forward = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm_backward = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.int_layer = nn.Linear(hidden_dim * 4, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, num_classes)
        self.embedding_layer = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

    def forward(self, a, b):

        a, b, a_rev, b_rev = fix_padding(a, b)
        a_embed = self.embedding_layer(a)
        b_embed = self.embedding_layer(b)
        a_rev_embed = self.embedding_layer(a_rev)
        b_rev_embed = self.embedding_layer(b_rev)

        a_lstm_output, (a_lstm_h, a_lstm_c) = self.lstm_forward(a_embed)
        b_lstm_output, (b_lstm_h, b_lstm_c) = self.lstm_forward(b_embed)

        a_rev_lstm_output, (a_rev_lstm_h, a_rev_lstm_c) = self.lstm_backward(a_rev_embed)
        b_rev_lstm_output, (b_rev_lstm_h, b_rev_lstm_c) = self.lstm_backward(b_rev_embed)

        cell_states = torch.cat((a_lstm_c, a_rev_lstm_c, b_lstm_c, b_rev_lstm_c), dim=2).squeeze(dim=0)
        int_output = nn.ReLU()(self.int_layer(cell_states))
        output = self.out_layer(int_output)

        return output

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)

def run_snli(model):

    # Set hyperparameters
    num_layers = 1
    num_classes = 3
    num_epochs = 10

    dataset = load_dataset("snli")
    glove = pd.read_csv(
        "./data/glove/glove.6B.50d.txt", sep=" ", quoting=3, header=None, index_col=0
    )

    # The glove embeddings are just the numpy array of the dataframe
    glove_embeddings = glove.values

    # The embedding dimension is the width of the glove table
    emb_dim = glove.shape[1]

    # Create embeddings dict map
    emb_dict = {word: glove_embeddings[i, :] for i, word in enumerate(glove.index)}

    # Filter data points with missing labels
    train_filtered = dataset["train"].filter(lambda ex: ex["label"] != -1)
    valid_filtered = dataset["validation"].filter(lambda ex: ex["label"] != -1).filter(lambda ex: len(ex["premise"]) > 0)
    test_filtered = dataset["test"].filter(lambda ex: ex["label"] != -1)

    # Create dataloaders
    dataloader_train = DataLoader(list(islice(train_filtered, 100)), batch_size=10)
    dataloader_valid = DataLoader(valid_filtered, batch_size=10)
    dataloader_test = DataLoader(test_filtered, batch_size=10)

    # code to make dataloaders
    print("Creating word_counts..")
    word_counts = build_word_counts(dataloader_train)
    print("Creating index_map..")
    index_map = build_index_map(word_counts)

    # training code
    embedding_matrix = create_embedding_matrix(index_map, emb_dict, emb_dim)

    model = UniLSTM(
        vocab_size=len(index_map),
        hidden_dim=emb_dim,
        num_layers=num_layers,
        num_classes=num_classes,
    )

    lr = 0.001

    # code to initialize optimizer, loss function
    optimizer = model.get_optimizer(lr=lr)
    loss_func = model.get_loss_function()

    n = 0
    running_loss = 0
    for epoch in range(num_epochs):
        for batch in dataloader_train:
            premises, hypotheses, labels = batch.values()
            premises = tokenize(premises)
            hypotheses = tokenize(hypotheses)

            batch_premises = tokens_to_ix(premises, index_map)
            batch_hypotheses = tokens_to_ix(hypotheses, index_map)

            # 1. Apply the RNN to the incoming sequence
            predictions = model.forward(batch_premises, batch_hypotheses)

            # 2. Use the loss function to calculate the loss on the model’s output
            current_loss = loss_func(predictions, labels)

            # 3. Zero the gradients of the optimizer
            optimizer.zero_grad()

            # 4. Perform a backward pass (calling .backward())
            current_loss.backward()

            # 5. Step the weights of the model via the optimizer (.step())
            optimizer.step()

            # 6. Add the current loss to the running loss
            running_loss += current_loss

        # Evaluate the model
        acc = evaluate(model, dataloader_valid, index_map)

        print(f"Epoch: {epoch} - Accuracy: {acc}")


def run_snli_lstm():
    model_class = (
        ""
    )  # fill in the classs name of the model (to initialize within run_snli)
    run_snli(model_class)


def run_snli_bilstm():
    model_class = (
        ""
    )  # fill in the classs name of the model (to initialize within run_snli)
    run_snli(model_class)


if __name__ == "__main__":

    # run_char_rnn()
    # run_char_lstm()
    run_snli_lstm()
    # run_snli_bilstm()

    print("Done!")
