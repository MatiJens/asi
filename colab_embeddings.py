import os
import torch
import glob
from Bio import SeqIO
from tqdm import tqdm
from google.colab import drive
from esm.models.esmc import ESMC

drive.mount("/content/drive")
BASE_PATH = "/content/drive/My Drive/Magisterka"
DIRECTORIES = [
    os.path.join(BASE_PATH, "data/positive_train"),
    os.path.join(BASE_PATH, "data/positive_val"),
    os.path.join(BASE_PATH, "data/positive_test"),
    os.path.join(BASE_PATH, "data/negative_train"),
    os.path.join(BASE_PATH, "data/negative_val"),
    os.path.join(BASE_PATH, "data/negative_test"),
]


MODEL_NAME = "esmc_600m"
BATCH_SIZE = 8
MAX_LENGTH = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sequences_from_dir(directory):
    """Load *.fasta/*.fa files from directory"""
    seq_list = []
    files = glob.glob(os.path.join(directory, "*.fasta")) + glob.glob(
        os.path.join(directory, "*.fa")
    )

    for filepath in files:
        try:
            for record in SeqIO.parse(filepath, "fasta"):
                seq_list.append(
                    {"id": record.id, "seq": str(record.seq), "len": len(record.seq)}
                )
        except Exception as e:
            print(f"Error during reading file: {filepath}: {e}")

    return seq_list


def save_shard(embeddings_dict, directory, shard_idx):
    """Save part of embeddings to *.pt file. Embeddings are saved in parts due to reduction of RAM usage."""
    output_path = os.path.join(directory, f"esmc_embeddings_shard_{shard_idx}.pt")
    torch.save(embeddings_dict, output_path)


def generate_embeddings():
    try:
        model = ESMC.from_pretrained(MODEL_NAME, device=DEVICE)
    except Exception as e:
        print(f"Error during model loading: {e}")
        return

    model.eval()

    for directory in DIRECTORIES:
        if not os.path.exists(directory):
            print(f"Missing directory: {directory}")
            continue

        raw_data = load_sequences_from_dir(directory)
        if not raw_data:
            continue

        raw_data.sort(key=lambda x: x["len"], reverse=True)

        embeddings_buffer = {}
        shard_counter = 0
        SAVE_EVERY = 500

        for i in tqdm(
            range(0, len(raw_data), BATCH_SIZE), desc=f"Przetwarzanie {directory}"
        ):
            batch = raw_data[i : i + BATCH_SIZE]
            seqs = [item["seq"][:MAX_LENGTH] for item in batch]
            ids = [item["id"] for item in batch]

            try:
                with torch.no_grad():
                    input_ids = model._tokenize(seqs).to(DEVICE)

                    output = model(input_ids)
                    batch_embeddings = output.embeddings

                for j, seq_id in enumerate(ids):
                    pad_idx = model.tokenizer.pad_token_id

                    valid_mask = input_ids[j] != pad_idx

                    seq_embedding = (
                        batch_embeddings[j][valid_mask].to(dtype=torch.float16).cpu()
                    )

                    embeddings_buffer[seq_id] = seq_embedding

                del input_ids, output, batch_embeddings

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM Error on batch {i}. Try reduce BATCH_SIZE parameter.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            if len(embeddings_buffer) >= SAVE_EVERY:
                save_shard(embeddings_buffer, directory, shard_counter)
                embeddings_buffer = {}
                shard_counter += 1
                torch.cuda.empty_cache()

        if embeddings_buffer:
            save_shard(embeddings_buffer, directory, shard_counter)

    print("Generation of embeddings finished")


if __name__ == "__main__":
    generate_embeddings()
