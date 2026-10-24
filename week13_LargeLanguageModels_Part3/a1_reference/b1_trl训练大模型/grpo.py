from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

dataset = load_dataset("trl-lib/tldr", split="train")

# Dummy reward function: count the number of unique characters in the completions
# completions代表模型生成的结果，其他字段可以从数据中传入，自己设计reward函数
def reward_num_unique_chars(completions, **kwargs):  # ✍️一句话生成多少个候选路径
    return [len(set(c)) for c in completions]  # ✍️每个句子中的每个字符的数量之和，字符数量越多奖励越高否则越低

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_num_unique_chars,  # 可以使用列表传入多个
    train_dataset=dataset,
    # args = GRPOConfig(output_dir="xx")
)
trainer.train()

#trainer.save_model()
