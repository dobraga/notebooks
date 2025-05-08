import asyncio
from logging import basicConfig, getLogger
from typing import Literal, TypedDict

from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.llms.google_genai import GoogleGenAI
from pydantic import BaseModel, Field

load_dotenv()
basicConfig(level="INFO")
logger = getLogger(__name__)
getLogger("google_genai").setLevel("ERROR")
getLogger("httpx").setLevel("ERROR")

MessageFlow = Literal["emotional", "logical"]


class MessageClassifier(BaseModel):
    message_type: MessageFlow = Field(..., description="'emotional' or 'logical'")


class MessageClassifierEvent(Event):
    message: str
    message_type: MessageFlow


# State schema
class ChatState(TypedDict):
    messages: list[dict[str, str]]
    message_type: MessageFlow | None


# Workflow definition
class ChatbotWorkflow(Workflow):
    classifier_llm = GoogleGenAI(model="gemini-1.5-flash").as_structured_llm(
        MessageClassifier
    )
    llm = GoogleGenAI(model="gemini-2.0-flash")

    @step
    async def classify(self, event: StartEvent) -> MessageClassifierEvent:
        message = event.content
        prompt = [
            ChatMessage.from_str(
                "Classify the user message as 'emotional' or 'logical'.",
                role="system",
            ),
            ChatMessage.from_str(message, role="user"),
        ]
        classifier = self.classifier_llm.chat(prompt).raw.message_type
        logger.info(f"Classified message as {classifier}")
        return MessageClassifierEvent(message=message, message_type=classifier)

    @step
    async def respond(self, event: MessageClassifierEvent) -> StopEvent:
        message = event.message
        message_type = event.message_type

        if message_type == "emotional":
            sys_msg = "You are a compassionate therapist. Focus on empathy, validations, and reflective questions."
        else:
            sys_msg = "You are a logical assistant. Provide concise, fact-based answers without addressing emotions."

        prompt = [
            ChatMessage.from_str(sys_msg, role="system"),
            ChatMessage.from_str(message, role="user"),
        ]
        reply = self.llm.chat(prompt).message.blocks[-1].text
        return StopEvent(result=reply)


async def run_chatbot():
    wf = ChatbotWorkflow()
    print("[Type 'exit' to quit]")

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
            break

        result = await wf.run(content=user_input)
        print(f"Assistant: {result}")


# Runner
if __name__ == "__main__":
    import asyncio

    asyncio.run(run_chatbot())
