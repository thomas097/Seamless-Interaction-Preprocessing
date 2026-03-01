import io
import tarfile
import requests
from tqdm import tqdm

def list_dataset_batches(
        base_url: str = "https://huggingface.co/api/datasets/facebook/seamless-interaction/tree/main?recursive=1",
        extension: str = "tar",
        skip: list[str] = ['extras']
        ) -> list[str]:
    """
    Lists all data (batch) files on HuggingFace Hub.
    """
    response = requests.get(base_url)
    response.raise_for_status()

    urls = []
    for item in response.json():
        if item['type'] == 'file' and item['path'].endswith('.' + extension) and not any(token in item['path'] for token in skip):
            urls.append(item['path'])
            print(item['path'])

    return urls


def load_batch_from_hub(
        batch_url: str,
        base_url = f"https://huggingface.co/datasets/facebook/seamless-interaction/resolve/main/"
        ) -> tarfile.TarFile:
    """
    Loads batch from dataset as TarFile from HuggingFace Hub.
    """
    response = requests.get(base_url + batch_url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("Content-Length", 0))
    chunk_size = 1024 * 1024  # 1 MB

    buffer = io.BytesIO()

    with tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"Downloading {batch_url}",
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                buffer.write(chunk)
                pbar.update(len(chunk))

    buffer.seek(0)

    return tarfile.open(fileobj=buffer, mode="r:*")


def load_batch_local(
        filepath: str
        ) -> tarfile.TarFile:
    """
    Loads batch from dataset as TarFile from local .tar file.
    As the complete dataset is quite large, this function is used for development only.
    """
    with open(filepath, mode='rb') as file:
        return tarfile.open(fileobj=file, mode="r:*")


if __name__ == '__main__':
    list_dataset_batches()
