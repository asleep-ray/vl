# Training a vision-language model

## Step 1. Dataset download

- Train dataset: [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/download)
- Image Download script: [
CLIP_prefix_caption](https://github.com/rmokady/clip_prefix_caption) -> `parse_conceptual.py`

## Step 2. Preparation of dataset

Below script generate dataset that has `processed_image` (i.e., image vector), `caption`, and `labels` (i.e., tokenized caption).
```bash
./prepare_dataset.sh
```

Note: Downloading and processing the full dataset is time-consuming. For this project, a subset of the Conceptual Captions dataset is used, split into training and validation sets in an 8:2 ratio.

- Train set: 8800 samples
- Valid set: 2200 samples


## Step 3. Training

This trains the integrated model for generating captions from images.
Integrated model that combines CLIP and FLAN-T5 models as the encoder and decoder, respectively.
For training, the encoder CLIP model is frozen. The decoder FLAN-T5 model is trained with the LoRA module.

Train the model with below command.

```bash
python train.py
```



## Note

__Specify which files have been modified__


- 기존 파이썬 스크립트를 수정했다기 보다는 전반적으로 아래 내용들을 바탕으로 작성하였습니다.
- 아래 reference 1에 있는 실습 코드를 참고하였습니다.
- transformers 에 CLIPModel 및 T5ForConditionalGeneration 모델을 활용하였습니다.




## Reference
1. [Generative AI with Large Language Models](https://www.coursera.org/learn/generative-ai-with-llms/)
2. Scaling Instruction-Finetuned Language Models [pdf](https://arxiv.org/pdf/2210.11416v5)
