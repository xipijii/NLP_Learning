"""
差旅助手Agent示例 - 演示大语言模型的function call能力
展示从用户输入 -> 工具调用 -> 最终回复的完整流程
"""

import os
import json
from openai import OpenAI

# ================= 阿里云百炼 API key =================
API_key = "sk-x"

# =============== 报销标准数据（静态配置） ===============

# 城市等级映射
CITY_LEVEL = {
    # 一线城市
    "北京": 1, "上海": 1, "深圳": 1, "广州": 1, "杭州": 1,
    # 二线城市
    "成都": 2, "武汉": 2, "南京": 2, "苏州": 2, "重庆": 2, "西安": 2,
    "天津": 2, "长沙": 2, "青岛": 2, "宁波": 2, "无锡": 2,
    # 三线及以下（默认）
}

# 差旅标准配置
TRAVEL_STANDARDS = {
    1: {  # 一线城市
        "meal_allowance": 100,
        "accommodation_ceiling": 550
    },
    2: {  # 二线城市
        "meal_allowance": 80,
        "accommodation_ceiling": 400
    },
    3: {  # 三线及以下
        "meal_allowance": 60,
        "accommodation_ceiling": 300
    }
}

# 职级系数（对住宿标准的影响）
LEVEL_MULTIPLIER = {
    "普通员工": 1.0,
    "经理": 1.0,
    "总监": 1.3,   # +30%
    "VP": 1.5,     # +50%
    "P5": 1.0, "P6": 1.0, "P7": 1.0,
    "P8": 1.3, "P9": 1.3,
    "P10": 1.5
}

# 费用类型发票要求
INVOICE_REQUIREMENTS = {
    "住宿费": {"need_invoice": True, "need_approval": False},
    "机票": {"need_invoice": True, "need_approval": False},
    "火车票": {"need_invoice": True, "need_approval": False},
    "打车费": {"need_invoice": True, "need_approval": False},
    "地铁公交": {"need_invoice": False, "need_approval": False},
    "餐饮_招待": {"need_invoice": True, "need_approval": False},
    "餐饮_个人": {"need_invoice": False, "need_approval": False},
    "自驾": {"need_invoice": False, "need_approval": True}
}

# ==================== 工具函数定义 ====================
# 每个企业有自己不同的产品，需要企业自己定义

def get_travel_standard(city: str, employee_level: str = "普通员工", standard_type: str = "住宿"):
    """
    查询公司差旅标准（不同城市、职级标准不同）

    Args:
        city: 城市名称，如"北京"、"成都"
        employee_level: 员工职级，如"普通员工"、"总监"、"VP"
        standard_type: 标准类型，"住宿"或"餐饮"

    Returns:
        标准金额、适用条件的JSON字符串
    """
    # 获取城市等级
    city_level = CITY_LEVEL.get(city, 3)  # 未匹配的默认为三线城市 todo
    standards = TRAVEL_STANDARDS[city_level]

    # 获取职级系数
    multiplier = LEVEL_MULTIPLIER.get(employee_level, 1.0)

    if standard_type == "住宿":
        base_amount = standards["accommodation_ceiling"]
        final_amount = int(base_amount * multiplier)
        result = {
            "city": city,
            "city_level": _get_city_level_name(city_level),
            "employee_level": employee_level,
            "standard_type": "住宿",
            "base_standard": base_amount,
            "multiplier": multiplier,
            "final_standard": final_amount,
            "note": f"职级{employee_level}住宿标准上浮{(multiplier - 1) * 100:.0f}%" if multiplier > 1 else "标准住宿标准"
        }
    elif standard_type == "餐饮":
        final_amount = standards["meal_allowance"]
        result = {
            "city": city,
            "city_level": _get_city_level_name(city_level),
            "employee_level": employee_level,
            "standard_type": "餐饮补贴",
            "daily_amount": final_amount,
            "note": "餐补无需发票，按出差天数发放"
        }
    else:
        result = {"error": f"不支持的标准类型：{standard_type}，请使用'住宿'或'餐饮'"}

    return json.dumps(result, ensure_ascii=False)


def calculate_travel_allowance(
        start_city: str,
        end_city: str,
        days: float,
        employee_level: str = "普通员工",
        meal_allowance_per_day: float = None,
        accommodation_allowance_per_day: float = None
):
    """
    计算差旅补贴（餐补+住宿补）

    Args:
        start_city: 出发城市
        end_city: 目的地城市
        days: 出差天数（可以是小数，如2.5表示2天半）
        employee_level: 员工职级
        meal_allowance_per_day: 每日餐补（如不提供则使用默认标准）
        accommodation_allowance_per_day: 每日住宿补（如不提供则使用默认标准）

    Returns:
        补贴明细和总额的JSON字符串
    """
    # 获取目的地城市等级和标准
    city_level = CITY_LEVEL.get(end_city, 3)
    standards = TRAVEL_STANDARDS[city_level]

    # 获取职级系数（住宿上浮）
    multiplier = LEVEL_MULTIPLIER.get(employee_level, 1.0)

    # 餐补（使用用户提供值或默认值）
    if meal_allowance_per_day is None:
        meal_allowance_per_day = standards["meal_allowance"]

    # 住宿补（使用用户提供值或默认值，并应用职级系数）
    if accommodation_allowance_per_day is None:
        accommodation_allowance_per_day = int(standards["accommodation_ceiling"] * multiplier)

    # 计算总补贴（住宿按天数-1计算，因为最后一天一般不产生住宿费）
    # 实际业务中：出差n天，住宿按n-1晚计算
    accommodation_nights = max(0, days - 1)  # todo

    total_meal = meal_allowance_per_day * days
    total_accommodation = accommodation_allowance_per_day * accommodation_nights

    result = {
        "trip_info": {
            "start_city": start_city,
            "end_city": end_city,
            "days": days,
            "employee_level": employee_level
        },
        "allowance_detail": {
            "meal_allowance_per_day": meal_allowance_per_day,
            "accommodation_allowance_per_day": accommodation_allowance_per_day,
            "meal_total": round(total_meal, 2),
            "accommodation_total": round(total_accommodation, 2),
            "accommodation_nights": accommodation_nights
        },
        "total_allowance": round(total_meal + total_accommodation, 2),
        "note": f"餐补{days}天共计{total_meal}元（无需发票），住宿{accommodation_nights}晚共计{total_accommodation}元（需提供发票）"
    }

    return json.dumps(result, ensure_ascii=False)


def check_expense_limit(
        expense_type: str,
        city: str,
        amount: float,
        employee_level: str = "普通员工"
):
    """
    根据公司差旅标准，检查某项费用是否超标  todo 没有按天算吗？

    Args:
        expense_type: 费用类型（住宿费/打车费/餐饮_招待等）
        city: 所在城市
        amount: 实际费用金额
        employee_level: 员工职级

    Returns:
        是否超标、标准金额、超出金额的JSON字符串
    """
    city_level = CITY_LEVEL.get(city, 3)
    standards = TRAVEL_STANDARDS[city_level]
    multiplier = LEVEL_MULTIPLIER.get(employee_level, 1.0)

    # 住宿费有标准
    if expense_type == "住宿费":
        standard_amount = int(standards["accommodation_ceiling"] * multiplier)
    else:
        # 交通费、餐饮招待等暂时没有严格上限（可扩展）
        standard_amount = None

    if standard_amount is None:
        result = {
            "expense_type": expense_type,
            "amount": amount,
            "standard_amount": "无固定标准",
            "is_over_limit": False,
            "excess_amount": 0,
            "note": f"{expense_type}无固定超标标准，请按实际发生额报销"
        }
    elif amount <= standard_amount:
        result = {
            "expense_type": expense_type,
            "city": city,
            "amount": amount,
            "standard_amount": standard_amount,
            "is_over_limit": False,
            "excess_amount": 0,
            "note": f"费用未超标，可全额报销"
        }
    else:
        excess = amount - standard_amount
        excess_rate = (excess / standard_amount) * 100

        # 判断是否需要特殊审批
        need_approval = excess_rate > 20
        approval_note = "需部门总监审批" if need_approval else "直属上级审批即可"

        result = {
            "expense_type": expense_type,
            "city": city,
            "amount": amount,
            "standard_amount": standard_amount,
            "is_over_limit": True,
            "excess_amount": round(excess, 2),
            "excess_rate": round(excess_rate, 2),
            "need_approval": need_approval,
            "approval_note": approval_note,
            "note": f"超标{round(excess, 2)}元（超{round(excess_rate, 2)}%），{approval_note}"
        }

    return json.dumps(result, ensure_ascii=False)


def need_invoice(expense_type: str, amount: float = None, company_policy: str = "standard"):
    """
    判断某项费用是否需要提供发票

    Args:
        expense_type: 费用类型
        amount: 费用金额（可选，某些场景金额会影响发票要求）
        company_policy: 公司政策（默认standard）

    Returns:
        是否需要发票、原因的JSON字符串
    """
    requirement = INVOICE_REQUIREMENTS.get(expense_type)

    if requirement is None:
        result = {
            "expense_type": expense_type,
            "need_invoice": True,
            "need_approval": False,
            "reason": f"未知费用类型，默认需要发票",
            "note": "建议保留发票以备查验"
        }
    else:
        # 特殊规则：地铁公交小额可能不需要发票
        if expense_type == "地铁公交" and amount and amount < 50:
            result = {
                "expense_type": expense_type,
                "need_invoice": False,
                "need_approval": False,
                "reason": "地铁公交单次50元以下可使用支付记录截图代替发票",
                "note": "建议保留支付凭证"
            }
        elif expense_type == "自驾":
            result = {
                "expense_type": expense_type,
                "need_invoice": False,
                "need_approval": requirement["need_approval"],
                "reason": "自驾按里程补贴，需提供起止记录，无需油票发票",
                "note": "需填写自驾里程登记表"
            }
        else:
            result = {
                "expense_type": expense_type,
                "need_invoice": requirement["need_invoice"],
                "need_approval": requirement["need_approval"],
                "reason": "需要发票" if requirement["need_invoice"] else "无需发票（已包含在补贴中）",
                "note": "请妥善保管发票" if requirement["need_invoice"] else ""
            }

    return json.dumps(result, ensure_ascii=False)


def calculate_reimbursement(
        travel_allowance: dict = None,
        receipts: list = None
):
    """
    汇总报销总额

    Args:
        travel_allowance: 差旅补贴信息（从calculate_travel_allowance获取）
        receipts: 发票明细列表，每项包含[type, amount, has_invoice]

    Returns:
        总报销金额、需提供发票的金额、无需发票的补贴金额
    """
    total = 0
    invoice_required_total = 0
    no_invoice_total = 0
    receipt_details = []

    # 1. 处理差旅补贴（无需发票部分）
    if travel_allowance:
        allowance_total = travel_allowance.get("total_allowance", 0)
        total += allowance_total
        no_invoice_total += allowance_total
        receipt_details.append({
            "type": "差旅补贴（餐补+住宿补）",
            "amount": allowance_total,
            "has_invoice": False,
            "reimbursable": True
        })

    # 2. 处理发票报销
    if receipts:
        for receipt in receipts:
            expense_type = receipt.get("type", "其他")
            amount = receipt.get("amount", 0)
            has_invoice = receipt.get("has_invoice", True)
            is_approved = receipt.get("is_approved", True)

            # 检查是否需要发票
            invoice_check = json.loads(need_invoice(expense_type, amount))
            actually_need_invoice = invoice_check.get("need_invoice", True)

            # 判断是否可报销
            can_reimburse = True
            if actually_need_invoice and not has_invoice:
                # 发票缺失：可写说明按50%报销
                amount = amount * 0.5
                can_reimburse = True
                invoice_check["note"] = "发票缺失，按50%报销"

            if can_reimburse:
                total += amount
                if has_invoice and actually_need_invoice:
                    invoice_required_total += amount
                else:
                    no_invoice_total += amount

                receipt_details.append({
                    "type": expense_type,
                    "original_amount": receipt.get("amount", 0),
                    "reimbursable_amount": amount,
                    "has_invoice": has_invoice,
                    "need_invoice": actually_need_invoice,
                    "note": invoice_check.get("note", "")
                })

    result = {
        "summary": {
            "total_reimbursement": round(total, 2),
            "invoice_required_amount": round(invoice_required_total, 2),
            "no_invoice_allowance": round(no_invoice_total, 2)
        },
        "details": receipt_details,
        "note": f"需提供发票的金额：{invoice_required_total}元，无需发票的补贴：{no_invoice_total}元"
    }

    return json.dumps(result, ensure_ascii=False)


# ==================== 辅助函数 ====================

def _get_city_level_name(level: int) -> str:
    """获取城市等级名称"""
    level_map = {1: "一线城市", 2: "二线城市", 3: "三线及以下"}
    return level_map.get(level, "未知")


# ==================== 工具函数的JSON Schema定义 ====================
# 这部分会成为LLM的提示词的一部分

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_travel_standard",
            "description": "查询公司差旅标准，包括不同城市、职级的住宿标准和餐饮补贴标准",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海、成都、武汉"
                    },
                    "employee_level": {
                        "type": "string",
                        "description": "员工职级，可选值：普通员工、经理、总监、VP、P5、P6、P7、P8、P9、P10",
                        "default": "普通员工"
                    },
                    "standard_type": {
                        "type": "string",
                        "description": "标准类型，可选值：住宿、餐饮",
                        "enum": ["住宿", "餐饮"],
                        "default": "住宿"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_travel_allowance",
            "description": "计算差旅补贴，包括餐补和住宿补贴。根据出差天数、目的地城市和职级自动计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_city": {
                        "type": "string",
                        "description": "出发城市"
                    },
                    "end_city": {
                        "type": "string",
                        "description": "目的地城市"
                    },
                    "days": {
                        "type": "number",
                        "description": "出差天数，可以是小数（如2.5表示2天半）"
                    },
                    "employee_level": {
                        "type": "string",
                        "description": "员工职级",
                        "default": "普通员工"
                    },
                    "meal_allowance_per_day": {
                        "type": "number",
                        "description": "每日餐补金额（可选，不提供则使用默认标准）"
                    },
                    "accommodation_allowance_per_day": {
                        "type": "number",
                        "description": "每日住宿补贴金额（可选，不提供则使用默认标准）"
                    }
                },
                "required": ["start_city", "end_city", "days"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_expense_limit",
            "description": "检查某项费用是否超过公司差旅标准（1天），需要参考实际天数计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expense_type": {
                        "type": "string",
                        "description": "费用类型，如：住宿费、打车费、餐饮_招待",
                        "enum": ["住宿费", "打车费", "餐饮_招待", "机票", "火车票"]
                    },
                    "city": {
                        "type": "string",
                        "description": "费用发生城市"
                    },
                    "amount": {
                        "type": "number",
                        "description": "实际费用金额"
                    },
                    "employee_level": {
                        "type": "string",
                        "description": "员工职级",
                        "default": "普通员工"
                    }
                },
                "required": ["expense_type", "city", "amount"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "need_invoice",
            "description": "判断某项费用是否需要提供发票",
            "parameters": {
                "type": "object",
                "properties": {
                    "expense_type": {
                        "type": "string",
                        "description": "费用类型",
                        "enum": ["住宿费", "机票", "火车票", "打车费", "地铁公交", "餐饮_招待", "餐饮_个人", "自驾"]
                    },
                    "amount": {
                        "type": "number",
                        "description": "费用金额（可选）"
                    },
                    "company_policy": {
                        "type": "string",
                        "description": "公司政策",
                        "default": "standard"
                    }
                },
                "required": ["expense_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_reimbursement",
            "description": "汇总计算总报销金额，包括差旅补贴和各类发票报销",
            "parameters": {
                "type": "object",
                "properties": {
                    "travel_allowance": {
                        "type": "object",
                        "description": "从calculate_travel_allowance函数返回的差旅补贴信息"
                    },
                    "receipts": {
                        "type": "array",
                        "description": "发票明细列表，每项包含type(费用类型)、amount(金额)、has_invoice(是否有发票)、is_approved(是否已审批)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "amount": {"type": "number"},
                                "has_invoice": {"type": "boolean"},
                                "is_approved": {"type": "boolean"}
                            }
                        }
                    }
                },
                "required": []
            }
        }
    }
]

# ==================== Agent核心逻辑 ====================

# 工具函数映射
available_functions = {
    "get_travel_standard": get_travel_standard,
    "calculate_travel_allowance": calculate_travel_allowance,
    "check_expense_limit": check_expense_limit,
    "need_invoice": need_invoice,
    "calculate_reimbursement": calculate_reimbursement
}


def run_agent(user_query: str, query_history: str = None, api_key: str = None, model: str = "qwen-plus"):
    """
    运行Agent，处理用户查询
    
    Args:
        user_query: 用户输入的问题
        query_history: 历史输入
        api_key: API密钥（如果不提供则从环境变量读取）
        model: 使用的模型名称
    """
    # 初始化OpenAI客户端
    client = OpenAI(
        api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 初始化对话历史
    messages = [
        {
            "role": "system",
            "content": query_history  # todo
        },
        {
            "role": "user",
            "content": user_query  # ✍️用户输入的部分
        }
    ]

    print("\n" + "=" * 60)
    print("【用户问题】")
    print(user_query)
    print("=" * 60)

    # Agent循环：最多进行5轮工具调用  # ✍️模仿agentic system的编排
    max_iterations = 5
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- 第 {iteration} 轮Agent思考 ---")

        # 调用大模型
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,  # ✍️注入function call，是openAI自己的规定，要通过这种方式去注入
            tool_choice="auto"  # 让模型自主决定是否调用工具
        )

        response_message = response.choices[0].message

        # 将模型响应加入对话历史
        messages.append(response_message)

        # 检查是否需要调用工具
        tool_calls = response_message.tool_calls

        if not tool_calls:
            # 没有工具调用，说明模型已经给出最终答案
            print("\n【Agent最终回复】")
            print(response_message.content)
            print("=" * 60)
            return response_message.content

        # 执行工具调用
        print(f"\n【Agent决定调用 {len(tool_calls)} 个工具】")

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"\n工具名称: {function_name}")
            print(f"工具参数: {json.dumps(function_args, ensure_ascii=False)}")

            # 执行对应的函数
            if function_name in available_functions:
                function_to_call = available_functions[function_name]  # ✍️通过函数名映射到真正的函数
                function_response = function_to_call(**function_args)  # ✍️将解析出来的参数传入函数

                print(f"工具返回: {function_response[:200]}..." if len(
                    function_response) > 200 else f"工具返回: {function_response}")

                # 将工具调用结果加入对话历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                })
            else:
                print(f"错误：未找到工具 {function_name}")

    print("\n【警告】达到最大迭代次数，Agent循环结束")
    return "抱歉，处理您的请求时遇到了问题。"


def main():
    api_key = API_key
    model = "qwen-plus"
    query_history = ""
    query_list = ["我去北京出差3天，普通员工，住宿和餐补标准是多少？总共能报多少补贴？"
                  , "出差住酒店花了600元/晚，公司标准是400，超标了200，这200能报销吗？"
                  , "报销：高铁票500元（有票），打车80元（有票），餐费150元（无票），帮我算总共能报多少钱。"
                  , "自定义问题"]
    while True:
        template = ""
        for index, query in enumerate(query_list):
            template += "" + str(index + 1) + ". " + query + "\n"
        print(template)
        number = int(input("请选择编号："))
        if number == len(query_list):
            query = input("请输入您的问题：")
        else:
            query = query_list[number-1]
        query_history += "; " + query

        run_agent(user_query=query, query_history=query_history, api_key=api_key, model=model)

        con = input("是否继续？（y/n）")
        if con == "n":
            break


if __name__ == "__main__":
    main()
