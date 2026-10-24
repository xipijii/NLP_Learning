import json
import pandas as pd
from datasets import load_dataset
#将arrow格式数据转成jsonl格式
def arrow_to_jsonl(path, output_path):
    dataset = load_dataset(path)
    for key in ["train", "validation", "test"]:
        if key not in dataset:
            continue
        f = open(output_path + "-%s.jsonl"%key, "w")
        for item in dataset[key]:
            # print(item)
            f.write(json.dumps(item) + "\n")
        f.close()


    
#将jsonl格式数据转成arrow格式
def jsonl_to_arrow(input_path, output_path):
    df = pd.read_json(input_path, lines=True)
    df.to_parquet(output_path)

if __name__ == "__main__":
    arrow_to_jsonl("Capybara", "Capybara")
    arrow_to_jsonl("tldr", "tldr")
    
    jsonl_to_arrow("tldr-test.jsonl", "tldr-test.arrow")
    