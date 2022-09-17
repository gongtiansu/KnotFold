#!/usr/bin/env python
import os
import sys
import click
import torch
import torch.nn.functional as F
import numpy as np
from model.main_model import MainModel as Model

def load_model(chk_path):
    model = Model().eval()
    chk = torch.load(chk_path, map_location=torch.device("cpu"))
    parsed_dict = {}
    for k, v in chk["state_dict"].items():
        if k.startswith("module."):
            k = k[7:]
        parsed_dict[k] = v
    model.load_state_dict(parsed_dict)
    return model

@click.command()
@click.option("-i", "--fasta", help="Input sequence (fasta format)", required=True)
@click.option("-o", "--outfile", help="Output file", required=True)
@click.option("-r", "--reference", help="Reference network or not", is_flag=True)
@click.option("--cuda", is_flag=True, default=True)

def main(fasta, outfile, reference, cuda):
    model = load_model(os.path.join(os.path.dirname(__file__), "weights", "reference.pth" if reference else "prior.pth"))
    lines = open(fasta, "r").readlines()
    seq = lines[1].strip().upper()
    seq = "".join([_ if _ in "ACGU" else "N" for _ in seq])
    vocab = np.full(128, -1, dtype=np.uint8)
    vocab[np.array("NAUCG", "c").view(np.uint8)] = np.arange(len("NAUCG"))
    seq = vocab[np.array(seq, "c").view(np.uint8)]
    data = {"seq": torch.from_numpy(seq[None]).long()}
    if cuda:
        model = model.cuda()
        data = {k: v.cuda() for k, v in data.items()}
    with torch.no_grad():
        output = model(data, inference_only=True)
    logits = output["contact_logits"]
    prob = torch.softmax(logits, dim=-1)[0, :, :, 1]
    prob = (prob + prob.transpose(-1, -2)) / 2
    prob = prob.cpu().numpy()
    with open(outfile, "w") as fp:
        for i in range(prob.shape[0]):
            for j in range(prob.shape[0]):
                fp.write("%.10f" % prob[i][j])
            fp.write("\n")


if __name__ == "__main__":
    main()
