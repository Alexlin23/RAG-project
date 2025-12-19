from graph import build_graph

app = build_graph()

# 初始化 state（所有字段都要给）
initial_state = {
    "user_input": "",
    "intent": None,
    "response": None,
}

result = app.invoke(initial_state)

print(result)
