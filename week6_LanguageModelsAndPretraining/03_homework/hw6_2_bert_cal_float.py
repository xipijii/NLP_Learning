from transformers import BertModel


def cal_float():
    embedding_float_num, layer_float_num, pooler_float_num = 0, 0, 0

    hidden_size = 768
    # head_size = 12
    # attention_head_size = 64
    vocab_size = 21128
    layer_num = 12

    # embedding
    embedding_float_num += vocab_size * hidden_size
    embedding_float_num += 2 * hidden_size
    embedding_float_num += 512 * hidden_size
    # embedding_float_num += hidden_size * hidden_size + hidden_size
    embedding_float_num += hidden_size + hidden_size  # ✅

    # self-attention
    layer_float_num += 3 * (hidden_size * hidden_size + hidden_size)
    layer_float_num += hidden_size * hidden_size + hidden_size

    # add+normalize
    # layer_float_num += hidden_size * hidden_size + hidden_size
    layer_float_num += hidden_size + hidden_size  # ✅

    # feed forward
    # layer_float_num += hidden_size * (4 * hidden_size) + hidden_size
    layer_float_num += hidden_size * (4 * hidden_size) + 4 * hidden_size  # ✅
    layer_float_num += (4 * hidden_size) * hidden_size + hidden_size

    # add+normalize
    # layer_float_num += hidden_size * hidden_size + hidden_size
    layer_float_num += hidden_size + hidden_size  # ✅

    layer_float_num *= 12

    pooler_float_num += hidden_size * hidden_size + hidden_size

    print(f"embedding_float_num={embedding_float_num}, layer_float_num={layer_float_num}, pooler_float_num={pooler_float_num}")
    print(f"step2 = {embedding_float_num + layer_float_num}")
    print(f"step3 = {embedding_float_num + layer_float_num + pooler_float_num}")
    return embedding_float_num + layer_float_num + pooler_float_num


def cal_bert_float():
    total_amount = 0
    bert = BertModel.from_pretrained(
        r".../01_reference/bert-base-chinese",
        return_dict=False)
    state_dict = bert.state_dict()
    key_list = state_dict.keys()
    for key in key_list:
        weight = state_dict.get(key)
        # shape = weight.shape
        # size = shape if len(shape) == 1 else shape[0] * shape[1]
        # total_amount += int(size)
        total_amount += weight.numel()  # ✍️ PyTorch的内置方法，直接返回张量中元素的总数，无论张量是多少维度
        print(f"key = {key}, weight_shape = {weight.shape} weight_size = {weight.numel()}, total_amount = {total_amount}")
    return total_amount


if __name__ == '__main__':
    bert_num = cal_bert_float()
    cal_num = cal_float()
    print(f"bert_num = {bert_num}")
    print(f"cal_num = {cal_num}")
    print(f"二者是否相等：{bert_num == cal_num}, 差值（bert_num - cal_num）为 = {bert_num - cal_num}")
    # 结果：102,267,648
