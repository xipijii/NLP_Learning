from trl import SFTTrainer
from datasets import load_dataset

#https://github.com/huggingface/trl  trl库的github地址
#https://huggingface.co/docs/trl/dataset_formats  支持的各类dataset格式


dataset = load_dataset("trl-lib/Capybara", split="train")

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
)
trainer.train()


