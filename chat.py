import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def main():
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    print("🤖 Chatbot CLI (ketik 'exit' untuk keluar)\n")

    messages = [
        {"role": "system", "content": "Kamu adalah asisten yang ramah dan membantu."}
    ]

    while True:
        user_input = input("Kamu: ")

        if user_input.lower() in ("exit", "quit"):
            print("👋 Sampai jumpa!")
            break

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # murah & cepat untuk CLI
            messages=messages,
            temperature=0.7,
        )

        reply = response.choices[0].message.content
        print(f"\nBot: {reply}\n")

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
