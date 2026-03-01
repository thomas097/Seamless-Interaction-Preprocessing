import json
import tarfile

def get_transcripts_from_batch(batch: tarfile.TarFile) -> dict[str, list[list[tuple[str, float, float]]]]:
    transcripts = dict()

    for member in batch.getmembers():
        if member.name.endswith(".json"):
            try:
                contents_str = batch.extractfile(member).read().decode("utf-8") #type:ignore
                contents = json.loads(contents_str)
            except Exception as e:
                print(f"Unable to read transcript from {member.name}: {e}")

            data = []
            for metadata in contents['metadata:transcript']:
                utt_tokens = [[t['word'], t['start'], t['end']] for t in metadata['words']]
                data.append(utt_tokens)

            transcripts[member.name[:-5]] = data

    return transcripts