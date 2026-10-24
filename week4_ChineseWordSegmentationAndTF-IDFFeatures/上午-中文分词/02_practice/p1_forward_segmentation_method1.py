"""
✍️
1. 确认词表最大长度（窗格）
2. 在字符串中移动窗格
    窗格内有词表中的词，窗格右侧左切分
    没有词表中的词，窗格从右往左回退，匹配词表中的词/单字后，右侧切分
3. 重复2，直到结束
"""


# 读取词表进数据结构，确认词表最大长度
def load_dict(path):
    max_word_length = 0
    word_dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            word = line.split()[0]
            word_dict[word] = 0
            max_word_length = max(max_word_length, len(word))
    return word_dict, max_word_length


# 切割
def cut1(sentence, max_word_length, word_dict):
    sentence_length = len(sentence)
    cut_list = []
    left, right = 0, 0
    # for left in range(sentence_length):
    while left < sentence_length:
        right = left + max_word_length - 1
        while left < right:
            check_str = sentence[left: right]
            if check_str not in word_dict.keys():
                right -= 1
                # 🤔找不到的场景
                if left == right:
                    cut_list.append(check_str)
                    left += 1
            else:
                cut_list.append(check_str)
                left = right
    return cut_list


"""
参考代码逻辑 ‼️
1. 用字符串切片代替索引操作
    切割之后，原字符串可以直接截取掉已切割部分
2. 边界处理：右边界为窗格右边界与字符串右边界的最小值
3. 分词的添加：
    单字处理
    匹配后的分词处理
"""
def cut2(sentence, max_word_length, word_dict):
    cut_list = []
    while sentence != '':
        window_right = min(len(sentence), max_word_length)
        check_str = sentence[:window_right]
            # ✍️窗格内字符串，需要依此进行回退检查
        while check_str not in word_dict:
            if len(check_str) == 1:
                break
            check_str = check_str[:len(check_str) - 1]
        cut_list.append(check_str)
        sentence = sentence[len(check_str):]
    return cut_list


def main_test(sentence, cut_method):
    word_dict, max_word_length = load_dict("dict.txt")
    cut_list = cut_method(sentence, max_word_length, word_dict)
    print(cut_list)

# todo 读取数据，使用特定分词方法之后存储数据到特定位置
def main(cut_method, input_path, output_path):
    pass

if __name__ == '__main__':
    test_sentence = "今天你学习了多长时间？有什么具体产出？"
    cut_method = cut1
    # test_sentence = "测试字符串"
    main_test(test_sentence, cut_method)

    # cut_method = cut1
    # input_path = ""
    # output_path = ""
    # main(cut_method, input_path, output_path)
