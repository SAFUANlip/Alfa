import os
import json

import torch
from credit_scoring.pytorch_training import inference
from credit_scoring.dataset_preprocessing_utils import features
from credit_scoring.model import CreditsGRU, CreditsLSTM

from credit_scoring.get_device import get_device

TEST_DATA_PATH = "data_MliF/test_data/"
TEST_BUCKETS_PATH = "data_MliF/test_buckets_rnn"


def infer():
    device = get_device()
    dataset_test = sorted([os.path.join(TEST_BUCKETS_PATH, x) for x in os.listdir(TEST_BUCKETS_PATH)])

    with open("data_MliF/embedding_projections.json") as f_in:
        embedding_projections = json.load(f_in)

    model = CreditsLSTM(features, embedding_projections, bidirectional=True).to(device)

    path_to_checkpoints = "C:/Users/User/Python/Alfa/weights/LSTM/"
    model.load_state_dict(torch.load(os.path.join(path_to_checkpoints, "best_checkpoint.pt")))
    test_preds = inference(model, dataset_test, batch_size=128, device=device)
    test_preds.to_csv("LSTM_submission.csv", index=None)


if __name__ == "__main__":
    infer()