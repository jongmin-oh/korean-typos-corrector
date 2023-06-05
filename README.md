
# Korean Typos(Spelling) Corrector Using Deep Learning

## 한국어 맞춤법 교정기
 - ETRI-et5 모델을 기반으로 fine-tuning한 한국어 구어체 전용 맞춤법 교정기 입니다.
 - 바로 사용하실 분들은 밑에 예제 코드 참고해서 모델('j5ng/et5-typos-corrector') 다운받아 사용하실 수 있습니다.

## Base on PLM model(ET5)
 - ETRI(https://aiopen.etri.re.kr/et5Model)

## Base on Dataset
 - 모두의 말뭉치(https://corpus.korean.go.kr/request/reausetMain.do?lang=ko)
 - 맞춤법 교정 데이터

### 예시
|original|corrected|
|------|---|
|이런게 눔 ㄱ ㅣ찮아서 ㅠㅠ|이런 게 넘 귀찮아서 ㅠㅠ|
|어쩌다 가게되써|어쩌다 가게 됐어?|
|이따 얘기하쟈|이따 얘기하자|
|ㅋㅋㅋㅋㅋㅋ언넝 맞이해|ㅋㅋㅋㅋㅋㅋ 얼른 맞이해|
|그냥 일을안가르쳐주고|그냥 일을 안 가르쳐 주고|

 ## Data Preprocessing
  - 특수문자 제거 (쉼표) .(마침표) 제거
  - null 값("") 제거
  - 너무 짧은 문장 제거(길이 2 이하) 
  - 문장 내 &name&, name1 등 이름 태그가 포함된 단어 제거(단어만 제거하고 문장은 살림)
  - total : 318,882 쌍

***

## How to use
```python
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# T5 모델 로드
model = T5ForConditionalGeneration.from_pretrained("j5ng/et5-typos-corrector")
tokenizer = T5Tokenizer.from_pretrained("j5ng/et5-typos-corrector")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "mps:0" if torch.cuda.is_available() else "cpu" # for mac m1

model = model.to(device) 

# 예시 입력 문장
input_text = "아늬 진짜 무ㅓ하냐고"

# 입력 문장 인코딩
input_encoding = tokenizer("맞춤법을 고쳐주세요: " + input_text, return_tensors="pt")

input_ids = input_encoding.input_ids.to(device)
attention_mask = input_encoding.attention_mask.to(device)

# T5 모델 출력 생성
output_encoding = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=128,
    num_beams=5,
    early_stopping=True,
)

# 출력 문장 디코딩
output_text = tokenizer.decode(output_encoding[0], skip_special_tokens=True)

# 결과 출력
print(output_text) # 아니 진짜 뭐 하냐고.
```

***

## With Transformer Pipeline
```python
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline

model = T5ForConditionalGeneration.from_pretrained('j5ng/et5-typos-corrector')
tokenizer = T5Tokenizer.from_pretrained('j5ng/et5-typos-corrector')

typos_corrector = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    framework="pt",
)

input_text = "완죤 어이업ㅅ네진쨬ㅋㅋㅋ"
output_text = typos_corrector("맞춤법을 고쳐주세요: " + input_text,
            max_length=128,
            num_beams=5,
            early_stopping=True)[0]['generated_text']

print(output_text) # 완전 어이없네 진짜 ᄏᄏᄏᄏ.
```
