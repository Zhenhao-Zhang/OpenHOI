import os
from openai import OpenAI

def rewrite_to_high_level(sentence: str, model: str = "gpt-4o") -> str:

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = (
        "You are an expert in language abstraction. "
        "Rewrite the following sentence into a high-level, concise, and abstract version "
        "that captures the essence without low-level details:\n\n"
        f"Sentence: \"{sentence}\"\n\n"
        "High-level version:"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"GPT API fail: {e}")


if __name__ == "__main__":
    import sys


    if len(sys.argv) > 1:
        input_sentence = " ".join(sys.argv[1:])
    else:
        input_sentence = input("your sentense: ")


    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("no set up for OPENAI_API_KEY")

    try:
        high_level = rewrite_to_high_level(input_sentence)
        print("\n origin prompt")
        print(input_sentence)
        print("\n High-level instructions")
        print(high_level)
    except Exception as e:
        print(f"error: {e}")