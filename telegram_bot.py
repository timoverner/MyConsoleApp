import os
from enum import Enum
from typing import TypedDict, Optional
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

# =======================
# LLM (Claude)
# =======================
llm = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    temperature=0.6,
)

async def ask(prompt: str) -> str:
    """Генерирует ответ через Claude"""
    response = llm.invoke([
        SystemMessage(content=(
            "Ты полезный и вежливый Telegram-ассистент. "
            "Веди разговор естественно, задавай 1–2 вопроса за раз, "
            "не делай длинные инструкции, не упоминай что ты ИИ."
        )),
        HumanMessage(content=prompt),
    ])
    return response.content

# =======================
# Стадии
# =======================
class Stage(str, Enum):
    INIT = "init"
    GOAL = "goal"
    LEVEL = "level"
    LANGUAGE = "language"
    PAST_COURSES = "past_courses"
    RECOMMEND = "recommend"
    FEEDBACK = "feedback"
    DONE = "done"

class BotState(TypedDict):
    stage: Stage
    goal: Optional[str]
    level: Optional[str]
    language: Optional[str]
    past_courses: Optional[str]
    last_user_message: Optional[str]
    reply: Optional[str]

# =======================
# Узлы
# =======================
async def init_node(state: BotState) -> BotState:
    reply = await ask(
        "Привет! Я помогу подобрать онлайн-курс. "
        "Скажи, чему ты хочешь научиться или какой навык развить?"
    )
    return {**state, "stage": Stage.GOAL, "reply": reply}

async def goal_node(state: BotState) -> BotState:
    goal = state.get("last_user_message")
    reply = await ask(
        f"Отлично! Твоя цель — {goal}. "
        "Какой у тебя уровень опыта — начальный, средний или продвинутый?"
    )
    return {**state, "goal": goal, "stage": Stage.LEVEL, "reply": reply}

async def level_node(state: BotState) -> BotState:
    level = state.get("last_user_message")
    reply = await ask(
        f"Поняла, уровень — {level}. "
        "На каком языке предпочитаешь учиться — русский или английский?"
    )
    return {**state, "level": level, "stage": Stage.PAST_COURSES, "reply": reply}

async def past_courses_node(state: BotState) -> BotState:
    language_msg = (state.get("last_user_message") or "").lower()
    language = "en" if "англ" in language_msg else "ru"
    reply = await ask(
        "Спасибо! А расскажи, какие онлайн-курсы ты проходил ранее? "
        "Это поможет подобрать что-то подходящее."
    )
    return {**state, "language": language, "stage": Stage.RECOMMEND, "reply": reply}

async def recommend_node(state: BotState) -> BotState:
    past_courses = state.get("last_user_message")
    reply = await ask(
        f"Итак, твоя цель: {state['goal']}, уровень: {state['level']}, язык: {state['language']}. "
        f"Ты уже проходил: {past_courses}. "
        "Вот 1–3 подходящих курса для тебя с кратким объяснением."
    )
    return {**state, "past_courses": past_courses, "stage": Stage.FEEDBACK, "reply": reply}

async def feedback_node(state: BotState) -> BotState:
    reply = await ask(
        "Подошли ли рекомендации, или хочешь скорректировать цель или уровень?"
    )
    return {**state, "stage": Stage.DONE, "reply": reply}

async def done_node(state: BotState) -> BotState:
    return state

# =======================
# Граф
# =======================
graph = StateGraph(BotState)
graph.add_node("init", init_node)
graph.add_node("goal", goal_node)
graph.add_node("level", level_node)
graph.add_node("past_courses", past_courses_node)
graph.add_node("recommend", recommend_node)
graph.add_node("feedback", feedback_node)
graph.add_node("done", done_node)

graph.add_edge(START, "init")
graph.add_edge("init", "goal")
graph.add_edge("goal", "level")
graph.add_edge("level", "past_courses")
graph.add_edge("past_courses", "recommend")
graph.add_edge("recommend", "feedback")
graph.add_edge("feedback", "done")
graph.add_edge("done", END)

app_graph = graph.compile()

# =======================
# Telegram handlers
# =======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state: BotState = {
        "stage": Stage.INIT,
        "goal": None,
        "level": None,
        "language": None,
        "past_courses": None,
        "last_user_message": None,
        "reply": None,
    }

    # Показываем индикатор "печатает"
    typing_msg = await update.message.reply_text("Печатает ответ...")
    result = await app_graph.ainvoke(state)
    await typing_msg.delete()

    context.user_data["state"] = result
    await update.message.reply_text(result["reply"])

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state: BotState = context.user_data.get("state")
    if not state:
        await start(update, context)
        return

    state["last_user_message"] = update.message.text

    typing_msg = await update.message.reply_text("Печатает ответ...")
    result = await app_graph.ainvoke(state)
    await typing_msg.delete()

    context.user_data["state"] = result
    if result.get("reply"):
        await update.message.reply_text(result["reply"])

# =======================
# Run
# =======================
def main():
    app = Application.builder().token(os.environ["TELEGRAM_TOKEN"]).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
