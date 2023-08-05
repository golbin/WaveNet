import argparse
from pathlib import Path

import librosa
import soundfile


def main(data_dir, output_dir, extension, sample_rate=16000, trim_frame_length=2048):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_recordings = data_dir.glob(f"*{extension}")

    for audio_path in all_recordings:
        output_path = output_dir / audio_path.name
        if output_path.exists():
            continue
        print(audio_path.name)
        audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
        audio, _ = librosa.effects.trim(audio, frame_length=trim_frame_length)

        soundfile.write(
            output_path,
            audio,
            samplerate=sample_rate,
            # subtype=extension.upper()
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Parent dir of all recordings")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Parent dir of all recordings")
    parser.add_argument("--extension", type=str, default="ogg",
                        help="What kind of extension you want to analyse.")
    args = parser.parse_args()

    main(args.data_dir, args.output_dir, args.extension)
