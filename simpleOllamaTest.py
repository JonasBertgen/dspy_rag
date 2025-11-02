from langchain_ollama import ChatOllama


def main():

    llm = ChatOllama(model="mistral", temperature=0)
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to German. Translate the user sentence.",
        ),
        ("human", "love programming."),
    ]
    print(llm.invoke(messages))


if __name__ == "__main__":
    main()
