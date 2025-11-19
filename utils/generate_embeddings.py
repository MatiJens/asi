import torch
import numpy as np
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from Bio import SeqIO


def generate_embeddings(input_path: str, output_path: str, client: ESMC) -> None:
    """Generate embeddings from FASTA file and saves it as *.npy file"""
    embeddings_collection = {}
    print(f"Generating embedding for file: {input_path}")
    for record in SeqIO.parse(input_path, "fasta"):
        seq_id = record.id
        print(f"Now generating embedding for sequence: {seq_id}", end="...")
        protein = ESMProtein(str(record.seq))
        protein_tensor = client.encode(protein)
        with torch.no_grad():
            logits_output = client.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True),
            )
        raw_embeddings = logits_output.embeddings.detach().cpu()
        final_embedding = raw_embeddings[0, 1:-1, :].numpy()
        embeddings_collection[seq_id] = final_embedding
        print("finished")
    np.save(output_path, embeddings_collection, allow_pickle=True)
    print(f"Embedding of {input_path} saved to {output_path}")
