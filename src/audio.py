import io
import tarfile
import numpy as np
from math import gcd
from scipy.io import wavfile
from scipy.signal import resample_poly

def _resample(audio, orig_sr: int, target_sr: int):
    if orig_sr == target_sr:
        return audio

    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g

    return resample_poly(audio, up, down)


def get_audio_from_batch(batch: tarfile.TarFile, sr: int = 24_000) -> dict[str, np.ndarray]:
    audios = dict()

    for member in batch.getmembers():
        if member.name.endswith(".wav"):
            try:
                # Extract bytes and convert to ndarray
                wav_bytes = batch.extractfile(member).read() # type:ignore
                with io.BytesIO(wav_bytes) as bio:
                    sample_rate, data = wavfile.read(bio)
            except Exception as e:
                print(f"Unable to read waveform from {member.name}: {e}")

            if sample_rate != sr:
                data = _resample(data, orig_sr=sample_rate, target_sr=sr)

            audios[member.name[:-4]] = data

    return audios