
"""
🤔使用前缀字典进行正向最大匹配分词
逻辑
1. 基于词表划分词前缀和词，得到前缀字典（词前缀标0，词标1）
2. 从前向后进行查找
    词前缀则继续移动（如果窗口内的词是一个词前缀则继续扩大窗口）
    词则划分，然后对剩余字符串继续处理（如果窗口内的词不是一个词前缀，则记录已发现的词，并将窗口移动到词边界）
"""

# 基于词表划分词前缀和词，得到前缀字典
# def load_dict(path, word_dict):  # ❌
# ✍️没有词表，是一次加载的，如何获取？
#  获取到一个词，从左向右移动
#  如果窗格内在前缀字典里没有，则是0
#  否则就是1
def load_dict(path):
    cut_dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            word = line.split()[0]
            # right = 0
            # while right < len(word):
            #     check_str = word[:right + 1]
            #     # if check_str not in word_dict:
            #     #     cut_dict[check_str] = 0
            #     # else:
            #     #     cut_dict[check_str] = 1
            #     # right += 1
            #     if check_str not in cut_dict:
            #         cut_dict[check_str] = 0
            #         right += 1
            # cut_dict[word] = 1
            for i in range(1, len(word)):
                check_str = word[:i]
                if check_str not in cut_dict:
                    cut_dict[check_str] = 0
            cut_dict[word] = 1
    return cut_dict

# 正向查找切分
def cut1(sentence, cut_dict):
    cut_list = []
    while sentence != '':
        # left = 0
        # while left < len(sentence):  # ❌
        #     check_str = sentence[:left]
        #     if cut_dict[check_str] == 0:
        #         left += 1
        #     else:
        #         cut_list.append(check_str)
        #         sentence = sentence[len(check_str):]
        #         break
        for i in range(1, len(sentence)):
            check_str = sentence[:i]
            if check_str not in cut_dict or cut_dict[check_str] == 1:
                # 窗口是一个词 ✍️或 窗口没在字典中出现
                cut_list.append(check_str)
                sentence = sentence[len(check_str):]
                break
    return cut_list

def main(sentence, cut_method, path):
    cut_dict = load_dict(path)
    return cut_method(sentence, cut_dict)

if __name__ == '__main__':
    test_sentence = "王羲之草书《平安帖》共有九行"
    cut_method = cut1
    path = "dict.txt"

    cut_list = main(test_sentence, cut_method, path)
    print(cut_list)
