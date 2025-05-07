from logging import basicConfig, getLogger
from typing import Literal, TypedDict

from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage
from llama_index.llms.google_genai import GoogleGenAI
from pydantic import BaseModel, Field

logger = getLogger(__name__)
load_dotenv()

basicConfig(level="INFO")
getLogger("google_genai").setLevel("ERROR")
getLogger("httpx").setLevel("ERROR")

tf = Literal["emotional", "logical"]


class MessageClassifier(BaseModel):
    message_type: tf = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response.",
    )


classifier_llm = GoogleGenAI(model="gemini-1.5-flash").as_structured_llm(
    MessageClassifier
)

llm = GoogleGenAI(model="gemini-2.0-flash")


class State(TypedDict):
    messages: list[dict[str, str]]
    message_type: tf | None


def classify_message(state: State) -> tf:
    last = state["messages"][-1]

    prompt = [
        ChatMessage.from_str(
            "Classify the user message as either 'emotional' or 'logical'.",
            role="system",
        ),
        ChatMessage.from_str(**last),
    ]

    classifier = classifier_llm.chat(prompt).raw.message_type

    logger.info(f"Classified message as {classifier}")

    return classifier


def therapist_response(content: str) -> str:
    prompt = [
        ChatMessage.from_str(
            "You are a compassionate therapist. Focus on empathy, validating emotions, and asking thoughtful questions.",
            role="system",
        ),
        ChatMessage.from_str(content, role="user"),
    ]
    return llm.chat(prompt).message.blocks[-1].text


def logical_response(content: str) -> str:
    prompt = [
        ChatMessage.from_str(
            "You are a purely logical assistant. Provide concise, fact-based answers without addressing emotions.",
            role="system",
        ),
        ChatMessage.from_str(content, role="user"),
    ]
    return llm.chat(prompt).message.blocks[-1].text


def run_chatbot():
    state: State = {"messages": [], "message_type": None}
    print("[Type 'exit' to quit]")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            print("Assistant: Goodbye!")
            break

        state["messages"].append({"role": "user", "content": user_input})
        mtype = classify_message(state)
        state["message_type"] = mtype

        if mtype == "emotional":
            reply = therapist_response(user_input)
        else:
            reply = logical_response(user_input)

        state["messages"].append({"role": "assistant", "content": reply})
        print(f"Assistant: {reply}")


if __name__ == "__main__":
    run_chatbot()
