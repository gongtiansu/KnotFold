import os
import sys
import tempfile
import subprocess
import click
import numpy as np
import torch
import torch.nn.functional as F
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

def inference(fasta, weight, cuda):
    model = load_model(os.path.join(os.path.dirname(__file__), weight))
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
    return prob


def predict(fasta, cuda):
    here = os.path.dirname(__file__)
    with tempfile.TemporaryDirectory() as d:
        fgs = [[]for _ in range(5)]
        for i in range(5):
            weight = "weights/prior_"+str(i)+".pth"
            fgs[i] = inference(fasta, weight, cuda)
        fg = np.mean(np.array(fgs, dtype=np.float64), axis=0)
        bg = inference(fasta, "weights/reference.pth", cuda)
        with open(os.path.join(d, "prior.mat"), 'w') as fp:
            for i in range(fg.shape[0]):
                for j in range(fg.shape[0]):
                    fp.write("%.10f" % fg[i][j])
                    fp.write("\t")
                fp.write("\n")
        with open(os.path.join(d, "reference.mat"), 'w') as fp:
            for i in range(bg.shape[0]):
                for j in range(bg.shape[0]):
                    fp.write("%.10f" % bg[i][j])
                    fp.write("\t")
                fp.write("\n")

        mincostflowcmd = f"{here}/KnotFold_mincostflow {d}/prior.mat {d}/reference.mat"
        p = subprocess.run(mincostflowcmd, shell=True, capture_output=True)
        assert p.returncode == 0
        pairs = []
        for line in p.stdout.decode().split("\n"):
            if len(line) == 0:
                continue
            l, r = line.split()
            pairs.append((int(l), int(r)))
    return pairs

def write_bpseq(seq, pairs, outfile):
    bp = [-1 for _ in seq]
    for l, r in pairs:
        bp[l-1] = r-1
        bp[r-1] = l-1
    with open(outfile, "w") as fp:
        for i, k in enumerate(seq):
            fp.write("%d %s %d\n" % (i+1, k, bp[i]+1))

@click.command()
@click.option("-i", "--fasta", help="Input sequence (fasta format)", required=True)
@click.option("-o", "--outdir", help="Output dictionary", default="./")
@click.option("--cuda", is_flag=True, default=True)


def main(fasta, outdir, cuda):
    task = open(fasta, "r").read().split("\n")
    name, seq = task[0][1:], task[1].strip()
    pairs = predict(fasta, cuda)
    write_bpseq(seq, pairs, os.path.join(outdir, f"{name}.bpseq"))

if __name__ == "__main__":
    main()
