from logging import basicConfig, getLogger
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


basicConfig(level="INFO")
logger = getLogger(__name__)
load_dotenv()


class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response.",
    )


classifier_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash"
).with_structured_output(MessageClassifier)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None


def classify_message(state: State):
    last_message = state["messages"][-1]

    result = classifier_llm.invoke(
        [
            {
                "role": "system",
                "content": """Classify the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions
            """,
            },
            {"role": "user", "content": last_message.content},
        ]
    )
    logger.info(result.message_type)
    return {"message_type": result.message_type}


def router(state: State):
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}

    return {"next": "logical"}


def therapist_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {
            "role": "system",
            "content": """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked.""",
        },
        {"role": "user", "content": last_message.content},
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


def logical_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {
            "role": "system",
            "content": """You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers based on logic and evidence.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses.""",
        },
        {"role": "user", "content": last_message.content},
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


def setup_graph():
    graph_builder = StateGraph(State)

    graph_builder.add_node("classifier", classify_message)
    graph_builder.add_node("router", router)
    graph_builder.add_node("therapist", therapist_agent)
    graph_builder.add_node("logical", logical_agent)

    graph_builder.add_edge(START, "classifier")
    graph_builder.add_edge("classifier", "router")

    graph_builder.add_conditional_edges(
        "router",
        lambda state: state.get("next"),
        {"therapist": "therapist", "logical": "logical"},
    )

    graph_builder.add_edge("therapist", END)
    graph_builder.add_edge("logical", END)

    return graph_builder.compile()


def run_chatbot():
    graph = setup_graph()

    state = {"messages": [], "message_type": None}

    print("[Type 'exit' to quit]")
    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    run_chatbot()
