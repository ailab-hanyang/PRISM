from huggingface_hub import snapshot_download
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Download specific folders from the PRISM dataset on Hugging Face Hub.")
    parser.add_argument("--local_dir", type=str, required=True, help="The local directory where the dataset will be downloaded.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    snapshot_download(
        repo_id="NeurIPS-2026-PRISM/PRISM-Dataset", 
        repo_type="dataset", 
        local_dir=args.local_dir,
        resume_download=True       
    )