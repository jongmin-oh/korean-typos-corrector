import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import Trainer, TrainingArguments


import pandas as pd
from sklearn.model_selection import train_test_split

# T5 모델 다운로드
# https://aiopen.etri.re.kr/et5Model

# T5 모델 로드
# model = T5ForConditionalGeneration.from_pretrained("다운 받은 ET5 모델 경로")
# tokenizer = T5Tokenizer.from_pretrained("다운 받은 ET5 모델 경로")

model = T5ForConditionalGeneration.from_pretrained("j5ng/et5-base")
tokenizer = T5Tokenizer.from_pretrained("j5ng/et5-base")

df = pd.read_csv("./data/typos_datasets.csv", index_col=0)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 입력 문장의 맨 앞에 "맞춤법을 고쳐주세요: " 추가
train_df["input"] = "맞춤법을 고쳐주세요: " + train_df["original"]
val_df["input"] = "맞춤법을 고쳐주세요: " + val_df["original"]

# 출력 문장의 끝에 "." 추가
train_df["output"] = train_df["corrected"] + "."
val_df["output"] = val_df["corrected"] + "."

train_encodings = tokenizer(
    train_df["input"].tolist(), max_length=128, padding=True, truncation=True
)
train_labels_encodings = tokenizer(
    train_df["output"].tolist(), max_length=128, padding=True, truncation=True
)
val_encodings = tokenizer(
    val_df["input"].tolist(), max_length=128, padding=True, truncation=True
)
val_labels_encodings = tokenizer(
    val_df["output"].tolist(), max_length=128, padding=True, truncation=True
)


class SpellCorrectionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels_encodings):
        self.encodings = encodings
        self.labels_encodings = labels_encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels_encodings["input_ids"][idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


train_dataset = SpellCorrectionDataset(train_encodings, train_labels_encodings)
val_dataset = SpellCorrectionDataset(val_encodings, val_labels_encodings)

# Trainer 설정
training_args = TrainingArguments(
    output_dir="./outputs",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=32,
    num_train_epochs=8,
    weight_decay=0.01,
    save_strategy="epoch",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 학습 실행
trainer.train()
