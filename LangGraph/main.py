from graph import build_graph, get_checkpointer

graph = build_graph()

# 线程配置（用于状态记忆）
thread_config = {"configurable": {"thread_id": "novel_session"}}

# 使用上下文管理器来管理 checkpointer
with get_checkpointer() as checkpointer:
    # 编译 graph（带持久化 checkpointer）
    app = graph.compile(checkpointer=checkpointer)
    
    # 检查是否有保存的状态
    saved_state = app.get_state(thread_config)

    if saved_state.values and saved_state.values.get('chapter_progress', 1) != 1 and saved_state.next:
        # 有保存的状态且有待继续的节点
        print(f"检测到上次运行状态，从节点 {saved_state.next} 继续...")
        print(f"章节进度: {saved_state.values.get('chapter_progress', 1)}")
        result = app.invoke(None, thread_config)
    elif saved_state.values and saved_state.values.get('chapter_progress', 1) != 1 and not saved_state.next:
        # 有保存的状态但next为空（quit退出的情况），直接从节点1继续
        print(f"检测到上次运行状态（已正常退出），章节进度: {saved_state.values.get('chapter_progress', 1)}")
        # 使用 update_state 将下一个节点设置为入口点
        app.update_state(thread_config, saved_state.values, as_node="__start__")
        print("从节点1继续...")
        result = app.invoke(None, thread_config)
    else:
        # 无保存状态，从头开始
        print("无保存状态，从头开始...")
        initial_state = {
            "user_input": "",
            "first_time": True,
            "prompts_message": None,
            "response": None,
            "chapter_progress": 1,
        }
        result = app.invoke(initial_state, thread_config)
