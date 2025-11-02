import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import Example, collate_batch
from model import LSTMAttentionVancomycin


def _to_device(batch, device):
    moved = []
    for item in batch:
        if isinstance(item, torch.Tensor):
            moved.append(item.to(device))
        else:
            moved.append(item)
    return tuple(moved)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAttentionVancomycin(device=device).to(device)
    state_dict = torch.load("best_lstm_attention.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    loader = DataLoader(Example(), batch_size=8, collate_fn=collate_batch)
    records = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            predictions, _ = model(batch)
            labels = batch[2]
            lengths = batch[8].long()
            for idx in range(predictions.size(0)):
                valid_len = lengths[idx].item()
                preds = predictions[idx, :valid_len].cpu()
                trues = labels[idx, :valid_len].cpu()
                print(f"Patient {idx} predictions: {preds.tolist()}")
                print(f"Patient {idx} targets:     {trues.tolist()}")
                for t in range(valid_len):
                    records.append(
                        {
                            "patient_index": int(idx),
                            "time_index": t,
                            "prediction": float(preds[t]),
                            "label": float(trues[t]),
                        }
                    )
    if records:
        df = pd.DataFrame(records)
        df.to_csv("lstm_attention_predictions.csv", index=False)
        print("Saved detailed predictions to lstm_attention_predictions.csv")


if __name__ == "__main__":
    main()
