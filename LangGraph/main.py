from graph import build_graph

app = build_graph()

user_input = input("请输入你的问题: ")

# 初始化 state（所有字段都要给）
initial_state = {
    "user_input": user_input,
    "intent": None,
    "response": None,
}

result = app.invoke(initial_state)

print(result)
