
'''
任务型多轮对话系统
读取场景脚本完成多轮对话

编程建议：
1.先搭建出整体框架
2.先写测试用例，确定使用方法
3.每个方法不要超过20行
4.变量命名和方法命名尽量标准
'''

import json
import pandas as pd
import re
import os

class DialogueSystem:
    def __init__(self):
        self.load()
    
    def load(self):
        self.all_node_info = {}
        self.load_scenario("scenario-买衣服.json")
        self.load_scenario("scenario-看电影.json")
        self.load_slot_templet("slot_fitting_templet.xlsx")


    def load_scenario(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            scenario = json.load(f)
        scenario_name = os.path.basename(file).split('.')[0]
        for node in scenario:
            self.all_node_info[scenario_name + "_" + node['id']] = node
            if "childnode" in node:
                self.all_node_info[scenario_name + "_" + node['id']]['childnode'] = [scenario_name + "_" + x for x in node['childnode']]
            

    def load_slot_templet(self, file):
        self.slot_templet = pd.read_excel(file)
        #三列：slot, query, values
        self.slot_info = {}
        #逐行读取，slot为key，query和values为value
        for i in range(len(self.slot_templet)):
            slot = self.slot_templet.iloc[i]['slot']
            query = self.slot_templet.iloc[i]['query']
            values = self.slot_templet.iloc[i]['values']
            if slot not in self.slot_info:
                self.slot_info[slot] = {}
            self.slot_info[slot]['query'] = query
            self.slot_info[slot]['values'] = values
      
    def nlu(self, memory):
        memory = self.intent_judge(memory)
        memory = self.slot_filling(memory)
        return memory

    def is_collecting_slot(self, memory):
        return bool(memory.get("require_slot") and memory.get("active_node"))

    def intent_judge(self, memory):
        # 重听策略：槽位追问阶段保持当前节点，避免上下文丢失
        if self.is_collecting_slot(memory):
            memory["hit_node"] = memory["active_node"]
            memory["intent_score"] = 1.0
            return memory
        query = memory['query']
        max_score = -1
        hit_node = None
        for node in memory["available_nodes"]:
            score = self.calucate_node_score(query, node)
            if score > max_score:
                max_score = score
                hit_node = node
        if max_score < 0.1 and memory.get("active_node"):
            memory["hit_node"] = memory["active_node"]
            memory["intent_score"] = max_score
            return memory
        memory["hit_node"] = hit_node
        memory["intent_score"] = max_score
        return memory


    def calucate_node_score(self, query, node):
        #节点意图打分，算和intent相似度
        node_info = self.all_node_info[node]
        intent = node_info['intent']
        max_score = -1
        for sentence in intent:
            score = self.calucate_sentence_score(query, sentence)
            if score > max_score:
                max_score = score
        return max_score
    
    def calucate_sentence_score(self, query, sentence):
        #两个字符串做文本相似度计算。jaccard距离计算相似度
        query_words = set(query)
        sentence_words = set(sentence)
        intersection = query_words.intersection(sentence_words)
        union = query_words.union(sentence_words)
        return len(intersection) / len(union)

    def fill_one_slot(self, memory, slot, query):
        """仅当命中槽位正则时才填槽；未命中则留给重听策略处理。"""
        if slot in memory:
            return memory
        slot_values = self.slot_info[slot]["values"]
        match = re.search(slot_values, query, re.I)
        if match:
            memory[slot] = match.group()
        return memory

    def slot_filling(self, memory):
        query = memory['query']
        hit_node = memory.get("active_node") or memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        slots = node_info.get('slot', [])
        if self.is_collecting_slot(memory):
            slots = [memory["require_slot"]]
        for slot in slots:
            memory = self.fill_one_slot(memory, slot, query)
        return memory

    def dst(self, memory):
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        slot = node_info.get('slot', [])
        prev_slot = memory.get("require_slot")
        memory["require_repeat"] = False
        for s in slot:
            if s not in memory:
                memory["require_slot"] = s
                memory["require_repeat"] = prev_slot == s
                return memory
        memory["require_slot"] = None
        return memory

    def dpo(self, memory):
        memory["active_node"] = memory["hit_node"]
        if memory["require_slot"] is None:
            memory["policy"] = "reply"
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["available_nodes"] = node_info.get("childnode", [])
        elif memory.get("require_repeat"):
            memory["policy"] = "repeat"
            memory["available_nodes"] = [memory["hit_node"]]  # 重复上一节点
        else:
            memory["policy"] = "request"
            memory["available_nodes"] = [memory["hit_node"]]
        return memory

    def nlg(self, memory):
        if memory["policy"] == "reply":
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["response"] = self.fill_in_slot(node_info["response"], memory)
            memory = self.fill_pending_action(memory, node_info)
        else:
            memory.pop("pending_action", None)
            slot = memory["require_slot"]
            query = self.slot_info[slot]["query"]
            if memory["policy"] == "repeat":
                memory["response"] = f"没听清，请再说一次：{query}"
            else:
                memory["response"] = query
        return memory

    def fill_in_slot(self, template, memory):
        node = memory["hit_node"]
        node_info = self.all_node_info[node]
        for slot in node_info.get("slot", []):
            if slot in memory:
                template = template.replace(slot, str(memory[slot]))
        return template

    # 填充sql
    def fill_pending_action(self, memory, node_info):
        actions = node_info.get("action")
        if not actions:
            memory.pop("pending_action", None)
            return memory
        if isinstance(actions, list):
            memory["pending_action"] = [
                self.fill_in_slot(str(a), memory) for a in actions
            ]
        else:
            memory["pending_action"] = self.fill_in_slot(str(actions), memory)
        return memory

    def run(self, query, memory):
        '''
        query: 用户输入
        memory: 用户状态
        '''
        memory["query"] = query
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)
        return memory
        

if __name__ == '__main__':
    ds = DialogueSystem()
    # print(ds.all_node_info)
    print(ds.slot_info)

    memory = {"available_nodes" :["scenario-买衣服_node1" ,"scenario-看电影_node1"]} #用户状态
    while True:
        query = input("请输入：")
        # query = "你好"    
        memory = ds.run(query, memory)
        print(memory)
        print()
        response = memory['response']
        print(response)
        print("===========")
