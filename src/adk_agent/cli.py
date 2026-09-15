import asyncio
import sys

from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

from adk_agent.agent import root_agent

load_dotenv()


async def run_cli() -> None:
    print("=== ADK RAG Assistant CLI ===")
    print("Type your questions below. Type 'exit' or 'quit' to stop.\n")

    runner = InMemoryRunner(agent=root_agent)
    user_id = "default_user"
    session_id = "cli-session"

    # Initialize the session asynchronously inside run_cli
    await runner.session_service.create_session(
        user_id=user_id,
        session_id=session_id,
        app_name=runner.app_name,
    )

    while True:
        try:
            user_input = input("User > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            new_message = Content(
                role="user",
                parts=[Part.from_text(text=user_input)],
            )

            print("\nAgent > ", end="", flush=True)

            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=new_message,
            ):
                content = getattr(event, "content", None)
                if content and content.parts:
                    for part in content.parts:
                        text = getattr(part, "text", None)
                        if text:
                            print(text, end="", flush=True)

            print("\n")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            sys.exit(0)


def main() -> None:
    # Synchronous entrypoint that delegates to the async event loop
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
