# Remote Download Rules

## Rule

For Hugging Face content:

- do not direct-download by default
- use mirror relay first
- use the Python downloader in this repository instead of ad-hoc shell snippets

This is implemented in:

- [download_utils.py](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/datasets/download_utils.py)
- [download_remote_file.py](/home/moonxkj/Desktop/MyWork/Graduation_Project/experiment/datasets/download_remote_file.py)

## Default Behavior

- Hugging Face URLs are rewritten to mirror URLs first
- default mirror base:
  - `https://hf-mirror.com`
- environment overrides are supported:
  - `GRADPROJ_HF_MIRROR`
  - `HF_ENDPOINT`
  - `HF_MIRROR`
- direct Hugging Face fallback is disabled unless `--allow-direct-hf` is passed explicitly

## Example

```bash
conda run -n Graph --no-capture-output python3 experiment/datasets/download_remote_file.py \
  --url 'https://huggingface.co/datasets/<repo>/resolve/main/file.csv' \
  --output experiment/dataset/_downloads/file.csv \
  --hf-mirror 'https://hf-mirror.com'
```

## Notes

- Non-Hugging-Face URLs are downloaded directly by the same Python streaming layer.
- Dataset preparation scripts should call the shared downloader instead of embedding one-off Hugging Face download logic.
