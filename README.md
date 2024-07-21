# Insanely Fast Whisper Truss API
An API to transcribe audio with [OpenAI's Whisper Large v3](https://huggingface.co/openai/whisper-large-v3)! Powered by 🤗 Transformers, Optimum & flash-attn

Features:
* 🎤 Transcribe audio to text at blazing fast speeds
* 📖 Fully open source and deployable on any GPU cloud provider
* 🗣️ Built-in speaker diarization
* ⚡ Easy to use and API layer
* 📃 Async background tasks and webhooks
* 🔥 Optimized for concurrency and parallel processing
* ✅ Task management, cancel and status endpoints
* 🔒 Admin authentication for secure API access
* 🧩 Fully managed API available on [JigsawStack](https://jigsawstack.com/speech-to-text)

Based on [Insanely Fast Whisper API](https://github.com/JigsawStack/insanely-fast-whisper-api) project. Reworked to run using truss framework making it easier to deploy on Baseten cloud.

This project is focused on providing a deployable blazing fast whisper API with truss on Baseten cloud infra by using cheaper GPUs and less resources but getting the same results with a A100. 

With [Baseten](https://www.baseten.co/), I've set up the `config.yaml` file to easily deploy on their infra!


Here are some benchmarks we ran on Nvidia A10G - 16GB and Baseten GPU infra👇
| Optimization type    | Time to Transcribe (150 mins of Audio) |
|------------------|------------------|
| **large-v3 (Transformers) (`fp16` + `batching [24]` + `Flash Attention 2`)** | **~2 (*1 min 38 sec*)**            |
| **large-v3 (Transformers) (`fp16` + `batching [24]` + `Flash Attention 2` + `diarization`)** | **~3 (*3 min 16 sec*)**            |
| **large-v3 (Transformers) (`fp16` + `batching [24]` + `Flash Attention 2` + `baseten startup`)** | **~2 (*1 min 10 sec*)**            |
| **large-v3 (Transformers) (`fp16` + `batching [24]` + `Flash Attention 2` + `diarization` + `baseten startup`)** | **~3 (*2 min 48 sec*)**|

The estimated startup time for the Baseten machine with GPU and loading up the model is around ~20 seconds. The rest of the time is spent on the actual computation.

## Deploying to Baseten
- Follow the [setup guide](https://docs.baseten.co/quickstart#setup) to get Truss CLI installed and authenticated with Baseten API key
- Clone the project locally and open a terminal in the root
- run `truss push --publish --trusted` to deploy the model

Run the following if you want to set up speaker diarization:

To get the Hugging face token for speaker diarization you need to do the following:
1. Accept [`pyannote/segmentation-3.0`](https://hf.co/pyannote/segmentation-3.0) user conditions
2. Accept [`pyannote/speaker-diarization-3.1`](https://hf.co/pyannote/speaker-diarization-3.1) user conditions
3. Create an access token at [`hf.co/settings/tokens`](https://hf.co/settings/tokens).
4. Set the secret in your [Baseten account](https://app.baseten.co/settings/secrets) with the key `hf_access_token`


Your API should look something like this:

```
https://{model_id}.api.baseten.co/production/predict
```

## Deploying to other cloud providers
Check out the original project [Insanely Fast Whisper API](https://github.com/JigsawStack/insanely-fast-whisper-api) which provides a dockerized version that can be deployed on any cloud provider with GPU support.

## Fully managed and scalable API 
[JigsawStack](https://jigsawstack.com) provides a bunch of powerful APIs for various use cases while keeping costs low. This project is available as a fully managed API [here](https://jigsawstack.com/speech-to-text) with enhanced cloud scalability for cost efficiency and high uptime. Sign up [here](https://jigsawstack.com) for free!

### Endpoints

Learn more about calling Baseten's APIs [here](https://docs.baseten.co/invoke/quickstart)

#### Base URL
```
POST https://{model_id}.api.baseten.co/production/predict
```

Transcribe or translate audio into text
##### Body params (JSON)
| Name    | value |
|------------------|------------------|
| url (Required) |  URL of audio |
| task | `transcribe`, `translate`  default: `transcribe` |
| language | `None`, `en`, [other languages](https://huggingface.co/openai/whisper-large-v3) default: `None` Auto detects language
| batch_size | Number of parallel batches you want to compute. Reduce if you face OOMs. default: `64` |
| timestamp | `chunk`, `work`  default: `chunk` |
| diarise_audio | Diarise the audio clips by speaker. You will need to set hf_token. default:`false` |

#### Webhook URL
```
POST https://{model_id}.api.baseten.co/production/async_predict
```

##### Body params (JSON)
| Name    | value |
|------------------|------------------|
| model_input (Required) |  body params of above API |
| webhook_endpoint | callback url |


## JigsawStack
This project is part of [JigsawStack](https://jigsawstack.com) - A suite of powerful and developer friendly AI APIs for various use cases while keeping costs low. Sign up [here](https://jigsawstack.com) for free!