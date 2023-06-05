import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# T5 모델 로드
model = T5ForConditionalGeneration.from_pretrained("j5ng/et5-typos-corrector")
tokenizer = T5Tokenizer.from_pretrained("j5ng/et5-typos-corrector")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "mps:0" if torch.cuda.is_available() else "cpu" # for mac m1

model = model.to(device)


def infer(input_text):
    input_encoding = tokenizer("맞춤법을 고쳐주세요: " + input_text, return_tensors="pt")

    input_ids = input_encoding.input_ids.to(device)
    attention_mask = input_encoding.attention_mask.to(device)

    output_encoding = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128,
        num_beams=5,
        early_stopping=True,
    )
    output_text = tokenizer.decode(output_encoding[0], skip_special_tokens=True)
    return output_text


if __name__ == "__main__":
    print(infer("완죤 어이업ㅅ네진쨬ㅋㅋㅋ"))
