import os
import sys
import tempfile
import subprocess
import click

def predict(fasta):
    here = os.path.dirname(__file__)
    with tempfile.TemporaryDirectory() as d:
        priorcmd = f"{here}/KnotFold_nn.py -i {fasta} -o {d}/prior.mat"
        referencecmd = f"{here}/KnotFold_nn.py -i {fasta} -o {d}/reference.mat -r"
        mincostflowcmd = f"{here}/KnotFold_mincostflow {d}/prior.mat {d}/reference.mat"
        p = subprocess.run(priorcmd, shell=True)
        assert p.returncode == 0
        p = subprocess.run(referencecmd, shell=True)
        assert p.returncode == 0
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

def main(fasta, outdir):
    pairs = predict(fasta)
    task = open(fasta, "r").read().split("\n")
    name, seq = task[0][1:], task[1].strip()
    write_bpseq(seq, pairs, os.path.join(outdir, f"{name}.bpseq"))

if __name__ == "__main__":
    main()
