import runpod
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", use_auth_token="YOUR_HF_TOKEN")

def handler(job):
    audio = job["input"]["audio"]
    segments, info = model.transcribe(audio, word_timestamps=True)
    result_segments = []
    for s in segments:
        words = []
        if s.words:
            for w in s.words:
                words.append({
                    "word": w.word,
                    "start": float(w.start) if w.start is not None else None,
                    "end": float(w.end) if w.end is not None else None
                })
        result_segments.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": s.text,
            "words": words
        })
    diarization = pipeline(audio)
    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker
        })
    return {
        "text": " ".join([s["text"] for s in result_segments]).strip(),
        "segments": result_segments,
        "speaker_segments": speaker_segments,
        "detected_language": getattr(info, "language", None),
        "duration_seconds": None
    }

runpod.serverless.start({"handler": handler})
