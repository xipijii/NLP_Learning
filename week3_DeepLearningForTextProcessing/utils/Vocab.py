

def build_vocab():
    # 1. 创建词表 第一个位置：占位符（padding token）
    vocab = ["<pad>"]

    # 2. 添加"你/我/他"
    vocab.extend(["你", "我", "他"])

    # 3. 添加其他常用中文字符（约 50 个）
    chinese_chars = [
        "的", "了", "在", "是", "和", "有", "这", "个", "上", "中",
        "为", "人", "们", "以", "到", "就", "说", "要", "大", "小",
        "多", "好", "很", "都", "把", "着", "看", "想", "能", "可",
        "去", "来", "出", "进", "过", "还", "也", "但", "会", "自",
        "从", "而", "之", "用", "所", "使", "并", "如", "因", "哈"
    ]
    vocab.extend(chinese_chars)

    # 4. 添加英文字母（大小写）
    letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    vocab.extend(letters)  # 26 个小写字母

    # 5. 添加数字
    digits = [str(i) for i in range(10)]
    vocab.extend(digits)

    # 6. 补充一些标点符号
    punctuations = ["，", "。", "？", "！", "；", "：", "?", "（", "）", "~"]

    vocab.extend(punctuations)

    # 7. 确保总长度为 100，最后一个位置留给未知符 <unk>
    # 如果超过 100，截断；如果不足 100，补充一些常见字符
    while len(vocab) < 99:  # 留出 2 个位置给 pad 和 unk
        vocab.append(f"char_{len(vocab)}")

    # 截断到 99 个
    vocab = vocab[:99]

    # 8. 最后一个位置：未知符（unknown token）
    vocab.append("<unk>")

    # 9. 转换为字典：键为字符，值为索引位置
    vocab_dict = {char: idx for idx, char in enumerate(vocab)}

    return vocab_dict