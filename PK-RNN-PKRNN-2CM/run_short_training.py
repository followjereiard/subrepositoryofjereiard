import torch

if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, device=None, non_blocking=False, memory_format=torch.preserve_format: self
    torch.nn.Module.cuda = lambda self, device=None, non_blocking=False: self

from train import run_model

paramdict = {
    "real_vanco_to_feedback": True,
    "change_regularize": "square",
    "eta1_var": 0.12,
    "eta2_var": 0.149,
    "eta3_var": 0.416,
    "scale_change_eta": 100,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_model(
    inFile="pk_sheets_dataset.h5",
    device=device,
    paramdict=paramdict,
    batchsize=2,
    n_epoch=10,
    lr=0.001,
    patience=2,
)
