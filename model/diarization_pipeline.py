from model.diarize import (
    post_process_segments_and_transcripts,
    diarize_audio,
    preprocess_inputs,
)


def diarize(diarization_pipeline, file_name, outputs):
    inputs, diarizer_inputs = preprocess_inputs(inputs=file_name)
    segments = diarize_audio(diarizer_inputs, diarization_pipeline)
    return post_process_segments_and_transcripts(
        segments, outputs["chunks"], group_by_speaker=False
    )
