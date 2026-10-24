import json
import re

"""
function
1. 加载、拼接文本数据
2. 构建词表：（直至达到预期的最终词表大小）
将文本进行utf-8编码
    获取二元组，按照bpe思想统计每个二元组出现的次数
    获取出现次数最多的二元组，并将其替换
3. tokenizer
4. decoding
"""

def load_corpus(corpus_path):
    corpus = ""
    with open(corpus_path, encoding="utf8") as f:
        for line in f:
            try:
                line_fixed = re.sub(r'(?<=: ")(.*?)(?="(?:,|\s*\}))',
                                    lambda m: m.group(0).replace('"', '\\"'), line.strip())
                obj = json.loads(line_fixed)
                corpus += obj["question"] + obj["answer"]
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON解析错误，跳过该行: {e}")
                print(f"   问题行内容: {repr(line[:100])}")
    return corpus

def build_vocab(corpus, vocab_size):
    utf8_tokens = corpus.encode("utf-8")
    ids = list(map(int, utf8_tokens))

    merges = {}  # ‼️二元组是词表的最小单位？
    merge_count = vocab_size - 256
    for i in range(merge_count):
        pairs = get_pair(ids)
        top_pair = max(pairs, key=pairs.get)
        idx = 256 + i
        # print(f"merging {top_pair} into a new token {idx}")
        ids = substitute(ids, top_pair, idx)
        merges[top_pair] = idx

    return merges

def get_pair(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def substitute(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def decoding(vocab, ids):  # ‼️
    # 原始字节:  [104, 101, 108, 108, 111]
    #                ↓ merge (104,101) → 263
    # 合并后:    [263, 108, 108, 111]
    # 263 代表的不是一个字符，而是两个字节 b'h' + b'e' 的组合。
    #   所以必须先把每个 ID 还原回它对应的原始字节片段，拼成完整字节串，最后再统一做 UTF-8 解码：
    # 263 → b'he'   # 108 → b'l'    # 108 → b'l'    # 111 → b'o'
    # 拼接 → b'hello' # 解码 → 'hello'
    # 中文应该是这种格式：b'\xe4\xbd'
    bytes_vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in vocab.items():
        bytes_vocab[idx] = bytes_vocab[p0] + bytes_vocab[p1]
    # 用空字节串作为分隔符，将一个可迭代对象中的所有 bytes 拼接在一起
    tokens = b"".join(bytes_vocab[idx] for idx in ids)
    text = tokens.decode("utf-8")
    return text


def tokenize(vocab, input_str):
    utf8_tokens = input_str.encode("utf-8")
    ids = list(map(int, utf8_tokens))
    while len(ids) >= 2:  # 只要列表中还有至少两个 token，就继续尝试合并
        pairs = get_pair(ids)
        pair = min(pairs, key=lambda p: vocab.get(p, float("inf")))
        if pair not in vocab:
            break
        ids = substitute(ids, pair, vocab[pair])
    return ids

def main(corpus_path):
    vocab_size = 5000
    corpus = load_corpus(corpus_path)
    vocab = build_vocab(corpus, vocab_size)  # utf8化，merge
    while True:
        input_str = input("请输入测试文本：")
        ids = tokenize(vocab, input_str)  # utf8化，根据词表获取ids
        decoding_str = decoding(vocab, ids)
        print(f"初始文本：{input_str}，utf8_ids：{ids}，解码后文本：{decoding_str}")
        print(f"文本长度：{len(input_str)}，token长度：{len(ids)}")

if __name__ == '__main__':
    corpus_path = "train_data.json"
    main(corpus_path)
