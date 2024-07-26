import argparse
import fileinput

from typing import *

import torch
import torch.nn as nn
import torch.onnx as onnx
import torch.optim as optim

"""
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
"""


def _parse_moses(fp: str, limit: int=None) -> Tuple[List[str], List[str]]:
    train, test = [], []

    for i, line in enumerate(fileinput.input([fp])):
        line = str(line).replace('\n', '')
        line = line.split(',')
        if line[-1] == 'train':
            train.append(line[:-1][0])
        else:
            test.append(line[:-1][0])

        if limit is not None and i >= limit:
            fileinput.close()
            return train, test
    fileinput.close()
    return train, test


def _create_char_embeddings(data: List[str]) -> Dict:
    chars = set()
    matrix = {}
    for chem_smile in data:
        chars.update(set(chem_smile))

    for number, char in enumerate(chars):
        matrix.update({char: number + 1})  # 0 is reserved for padding

    return matrix


def _padding(data: List[int], pad_len: int, value: int = 0, pad_type='after') -> List[int]:
    pad_amount = pad_len - len(data)

    if pad_type == 'after':
        if pad_amount > 0:
            data.extend([value for _ in range(pad_amount)])

        if pad_amount < 0:
            data = data[:pad_len]

    if pad_type == 'before':
        data = [value for _ in range(pad_amount)] + data

    return data


def _encode_smile(smile: str, embedding_matrix: Dict, padding: bool = True, pad_len: int = None, value: int = 0) -> List[int]:
    encoded = [embedding_matrix[char] for char in smile]
    if padding:
        if pad_len is None:
            raise ValueError('pad_len must not be None')
        encoded = _padding(encoded, pad_len=pad_len, value=value)

    return encoded


def _one_hot_encode_data(data: List[List[int]], embedding_matrix: Dict) -> List[List[int]]:
    for smile_index, smile in enumerate(data):
        for number_index, number in enumerate(smile):
            one_hot = torch.zeros(len(embedding_matrix.keys()) + 1).tolist()
            one_hot[number] = 1

            smile[number_index] = one_hot
        data[smile_index] = smile
    return data


def find_max_len(data: List[List]) -> int:
    max_len = 0
    for lst in data:
        if len(lst) > max_len:
            max_len = len(lst)
    return max_len


def prepare_moses(fp: str='./data/moses.csv', limit: int=None,
                  pad_len: int=35, return_matrix: bool=False) -> torch.Tensor:
    smile_data = _parse_moses(fp, limit=limit)[0]
    char_embeddings = _create_char_embeddings(smile_data)

    for i, smile in enumerate(smile_data):
        smile_data[i] = _encode_smile(smile, char_embeddings, pad_len=pad_len)

    if return_matrix:
        return _one_hot_encode_data(smile_data, embedding_matrix=char_embeddings), char_embeddings
    return _one_hot_encode_data(smile_data, embedding_matrix=char_embeddings)


class GRU(nn.Module):

    def __init__(self, input_size: int, hidden_size: int,
                 batch_size: int, output_size: int, n_layers: int, bidirectional: bool=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.bidirectional = bidirectional

        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                          batch_first=True, num_layers=self.n_layers, bidirectional=self.bidirectional)

        self.l1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size, self.output_size)

        if self.bidirectional:
            self.l1 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
            self.l2 = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, x):
        hidden = self.init_hidden(self.batch_size)
        output, hidden = self.gru(x.view(1, 1, -1), hidden)

        dense = self.l1(output.view(1, -1))
        dense = self.l2(dense)
        return dense, hidden

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return torch.zeros(self.n_layers * 2, batch_size, self.hidden_size)
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)


def train(network: nn.Module, seq: torch.Tensor, trgt: torch.Tensor,
          criterion: Any, sequence_len: int=35, optimizer: Any=optim.Adam) -> float:

    optimizer = optimizer(network.parameters())

    network.train()
    network.zero_grad()
    train_loss = 0

    for i in range(sequence_len - 1):
        output, _ = network(seq[i])

        train_loss += criterion(output, torch.argmax(trgt[i].view(1, -1), dim=1))

    train_loss.backward()
    optimizer.step()

    return train_loss.item() / sequence_len


def evaluate(network: nn.Module, embedding_matrix: Dict, prime_str: str='CC',
             predict_len: int=35, temperature: float=1.0):

    network.eval()

    decoding_matrix = {0: ''}
    for key, value in embedding_matrix.items():
        decoding_matrix.update({value: key})

    predicted = prime_str
    prime_input = _one_hot_encode_data([_encode_smile(prime_str,
                                                      embedding_matrix, pad_len=35)], embedding_matrix)
    prime_input = torch.Tensor(prime_input)
    for i in range(len(prime_str)):
        network(prime_input[0][i])
    seq = torch.Tensor([prime_input[0][-1].tolist()])

    for i in range(predict_len):
        output, h = network(seq)

        choice = torch.multinomial(output.data.view(-1).div(temperature).exp(), 1)[0]

        predicted_char = decoding_matrix[choice.item()]
        predicted += predicted_char

        seq = _one_hot_encode_data([_encode_smile(predicted_char, embedding_matrix, pad_len=35)], embedding_matrix)
        seq = torch.Tensor(seq[0][i])

    return predicted


def main():
    x_data, char_matrix = prepare_moses(return_matrix=True, limit=DATA_LIMIT)

    model = GRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, batch_size=BATCH_SIZE,
                output_size=OUTPUT_SIZE, n_layers=N_LAYERS, bidirectional=BIDIRECTIONAL)

    global MODEL_NAME
    if MODEL_NAME is None:
        MODEL_NAME = 'GRU-Model-{}-{}.onnx'.format(N_LAYERS, HIDDEN_SIZE)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(1):

        for iteration, element in enumerate(x_data):
            element = torch.Tensor([element])

            sequence, target = element[0][:-1], element[0][1:]

            loss = train(model, sequence, target, criterion=criterion)

            smile_str = evaluate(model, char_matrix, prime_str='CC(=O)')
            print('{}\t Loss: {}\n'.format(smile_str, loss))

            if iteration >= ITERATION_LIMIT:
                break

    onnx.export(model, sequence[0], MODEL_NAME)
    print(evaluate(model, prime_str='CC(=O)', embedding_matrix=char_matrix))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run GRU for molecular generation')

    parser.add_argument('-limit', '--limit', type=int,
                        default=10000, help='Limit for number of lines to read from data')
    parser.add_argument('-input_size', '--input_size', type=int, default=26, help='GRU input size')
    parser.add_argument('-hidden_size', '--hidden_size', type=int, default=150, help='GRU hidden size')
    parser.add_argument('-batch_size', '--batch_size', type=int, default=1, help='GRU batch size')
    parser.add_argument('-layers', '--layers', type=int, default=2, help='Number of GRU layers')
    parser.add_argument('-epochs', '--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('-bidirectional', '--bidirectional', type=bool, default=True, help='If GRU is bidirectional')
    parser.add_argument('-model_name', '--model_name', type=str, default=None, help='Saved model name')
    parser.add_argument('-iteration_limit', '--iteration_limit',
                        type=int, default=75, help='End training after a certain amount of '
                                                   'iterations within the epoch [RECOMMENDED]')
    args = parser.parse_args()

    DATA_LIMIT = args.limit
    INPUT_SIZE = args.input_size
    HIDDEN_SIZE = args.hidden_size
    BATCH_SIZE = args.batch_size
    OUTPUT_SIZE = INPUT_SIZE
    N_LAYERS = args.layers
    EPOCHS = args.epochs
    BIDIRECTIONAL = args.bidirectional
    MODEL_NAME = args.model_name

    ITERATION_LIMIT = args.iteration_limit  # Recommended due to convergence error

    main()
