import argparse

import numpy as np
import torch
from pnpl.datasets import LibriBrainCompetitionHoldout
from tqdm import tqdm

from speechclassifier import SpeechClassifier, SpeechClassifierSTFT
from train import convert_to_stft, normalize_per_sensor

SENSORS_SPEECH_MASK = [18, 20, 22, 23, 45, 120, 138, 140, 142, 143, 145,
                       146, 147, 149, 175, 176, 177, 179, 180, 198, 271, 272, 275]

def main(args):
    model = SpeechClassifierSTFT(mode="classification").to(args.device)
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    dataset = LibriBrainCompetitionHoldout(
        data_path='./datasets/submission',
        tmax=0.8,
        task="speech",
        stride=None
    )

    dataset_len = len(dataset)

    window_predictions = []

    for data, _ in tqdm(dataset):
        # data = data[SENSORS_SPEECH_MASK, :]
        data = data.to(args.device, non_blocking=True)
        data = data.unsqueeze(0)

        prediction_length = data.shape[2]

        if prediction_length < 200:
            prediction_length = data.shape[2]
            data = torch.nn.functional.pad(data, (0, 200 - data.shape[2]))

        data = convert_to_stft(data)
        data = normalize_per_sensor(data)

        with torch.no_grad():
            logits = model(data)

            predictions = torch.softmax(logits, dim=-1).squeeze(0)
            predictions = predictions[:prediction_length]

            window_predictions.extend(predictions[:,1].unsqueeze(1))

    dataset.generate_submission_in_csv(window_predictions, args.output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, default="weights/best.pt")
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--device", type=str, default="cuda")

    main(parser.parse_args())