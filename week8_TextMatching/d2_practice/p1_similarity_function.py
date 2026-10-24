import json
import numpy as np
import jieba

'''

包含编辑距离和jaccard距离的实现

'''

# 编辑距离
def editing_distance(string1, string2):
    matrix = np.zeros((len(string1) + 1, len(string2) + 1))
    for i in range(len(string1) + 1):
        matrix[i][0] = i
    for j in range(len(string2) + 1):
        matrix[0][j] = j
    for i in range(1, len(string1) + 1):
        for j in range(1, len(string2) + 1):
            if string1[i - 1] == string2[j - 1]:
                d = 0
            else:
                d = 1
            # ‼️ matrix[i - 1][j] + 1 删除    表示先将 S_1 的前 i-1 个字符变成 S_2 的前 j 个，然后删掉 S_1 的第 i 个字符
            # ‼️ matrix[i][j - 1] + 1 插入    表示先将 S_1 的前 i 个字符变成 S_2 的前 j-1 个，然后在末尾插入 S_2 的第 j 个字符
            # ‼️ matrix[i - 1][j - 1] + d 替换/不动     如果 S_1[i-1] == S_2[j-1]，则 d=0，如果不同，则 d=1（需要替换一次字符）
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
    edit_distance = matrix[len(string1)][len(string2)]
    return 1 - edit_distance / max(len(string1), len(string2))


# todo
def editing_distance2(string1, string2):
    matrix = np.zero(len(string1) + 1, len(string2) + 1)
    for i in range(len(string1) + 1):
        matrix[i][0] = i
    for j in range(len(string2) + 1):
        matrix[0][j] = j

    for i in range(1, len(string1) + 1):
        for j in range(1, len(string2) + 1):
            if string1[i - 1] == string2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i][j-1] + 1, matrix[i-1][j] + 1, matrix[i-1][j-1] + d)

    editing_distance2 = matrix[len(string1)][len(string2)]
    return 1 - editing_distance2 / max(len(string1), len(string2))


# jaccard距离
def jaccard_distance(string1, string2):
    words1 = set(string1)
    words2 = set(string2)
    distance = len(words1 & words2) / len(words1 | words2)
    return distance


if __name__ == "__main__":
    a = "abcd"
    b = "acde"
    print(editing_distance(a, b))
    print(jaccard_distance(a, b))