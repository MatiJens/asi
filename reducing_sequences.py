import random
from pathlib import Path
from typing import List
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def process_fa_files(directory_path: str, output_file: str) -> None:
    source_path = Path(directory_path)
    all_sequences: List[SeqRecord] = []

    fa_files = list(source_path.glob("*.fa"))

    for file_path in fa_files:
        sequences = list(SeqIO.parse(file_path, "fasta"))
        all_sequences.extend(sequences)

    total_count = len(all_sequences)
    sample_size = int(total_count * 0.002)

    random_subset = random.sample(all_sequences, sample_size)

    SeqIO.write(random_subset, output_file, "fasta")

    print(f"Total sequences found: {total_count}")
    print(f"Sequences in output file: {len(random_subset)}")


if __name__ == "__main__":
    process_fa_files("data/negative_val/", "data/negative_val/negative_val.fasta")
