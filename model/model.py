import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from model.diarization_pipeline import diarize
from pyannote.audio import Pipeline

model_id = "openai/whisper-large-v3-turbo"


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self.hf_token = self._secrets["hf_access_token"]
        self._model = None
        self.diarization_pipeline = None

    def load(self):

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        ).to("cuda:0")

        processor = AutoProcessor.from_pretrained(model_id)

        self._model = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16,
            device_map="auto",
            # model_kwargs=({"attn_implementation": "flash_attention_2"}),
        )

        self.diarization_pipeline = Pipeline.from_pretrained(
            checkpoint_path="pyannote/speaker-diarization-3.1",
            use_auth_token=self.hf_token,
        )

        self.diarization_pipeline.to(torch.device("cuda"))

    def predict(self, request: dict):
        try:
            url = request.get("url")
            task = request.get("task") or "transcribe"
            language = request.get("language") or "None"
            batch_size = request.get("batch_size") or 64
            chunk_length_s = request.get("chunk_length_s") or 30
            timestamp = request.get("timestamp") or "Chunk"
            diarise_audio = request.get("diarise_audio") or False

            generate_kwargs = {
                "task": task,
                "language": (
                    None if (language == "None" or language == None) else language
                ),
            }

            outputs = self._model(
                url,
                chunk_length_s=chunk_length_s,
                batch_size=batch_size,
                generate_kwargs=generate_kwargs,
                return_timestamps="word" if timestamp == "word" else True,
            )

            if diarise_audio:
                speakers_transcript = diarize(
                    self.diarization_pipeline,
                    url,
                    outputs,
                )
                outputs["speakers"] = speakers_transcript
            return outputs
        except Exception as e:
            print(e)
            return {"error": True, "detail": str(e)}
